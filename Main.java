import com.sun.net.httpserver.HttpExchange;

import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.Executors;

import java.util.function.BiFunction;

public class Main {
    static final int DIMS = 256;

    // =====================================================================
    // DATA TYPES
    // =====================================================================
    static class VectorItem {
        int id;
        String metadata;
        String category;
        float[] emb;

        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.emb = emb;
        }
    }

    interface DistFn extends BiFunction<float[], float[], Float> {
    }

    // =====================================================================
    // DISTANCE METRICS
    // =====================================================================
    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f)
            return 1.0f;
        return 1.0f - dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
    }

    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++)
            s += Math.abs(a[i] - b[i]);
        return s;
    }

    static DistFn getDistFn(String m) {
        if ("cosine".equals(m))
            return Main::cosine;
        if ("manhattan".equals(m))
            return Main::manhattan;
        return Main::euclidean;
    }

    static class DistResult implements Comparable<DistResult> {
        float dist;
        int id;

        DistResult(float dist, int id) {
            this.dist = dist;
            this.id = id;
        }

        public int compareTo(DistResult o) {
            int c = Float.compare(this.dist, o.dist);
            return c != 0 ? c : Integer.compare(this.id, o.id);
        }
    }

    // =====================================================================
    // BRUTE FORCE
    // =====================================================================
    static class BruteForce {
        List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem v) {
            items.add(v);
        }

        List<DistResult> knn(float[] q, int k, DistFn dist) {
            List<DistResult> r = new ArrayList<>(items.size());
            for (VectorItem v : items)
                r.add(new DistResult(dist.apply(q, v.emb), v.id));
            Collections.sort(r);
            if (r.size() > k)
                r = r.subList(0, k);
            return r;
        }

        void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }

    // =====================================================================
    // KD-TREE
    // =====================================================================
    static class KDNode {
        VectorItem item;
        KDNode left = null;
        KDNode right = null;

        KDNode(VectorItem v) {
            this.item = v;
        }
    }

    static class KDTree {
        KDNode root = null;
        int dims;

        KDTree(int d) {
            this.dims = d;
        }

        KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null)
                return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax])
                n.left = ins(n.left, v, d + 1);
            else
                n.right = ins(n.right, v, d + 1);
            return n;
        }

        void knn(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<DistResult> heap) {
            if (n == null)
                return;
            float dn = dist.apply(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().dist) {
                heap.add(new DistResult(dn, n.item.id));
                if (heap.size() > k)
                    heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knn(closer, q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().dist)
                knn(farther, q, k, d + 1, dist, heap);
        }

        void insert(VectorItem v) {
            root = ins(root, v, 0);
        }

        List<DistResult> knn(float[] q, int k, DistFn dist) {
            PriorityQueue<DistResult> heap = new PriorityQueue<>(Collections.reverseOrder());
            knn(root, q, k, 0, dist, heap);
            List<DistResult> r = new ArrayList<>();
            while (!heap.isEmpty())
                r.add(heap.poll());
            Collections.sort(r);
            return r;
        }

        void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem v : items)
                insert(v);
        }
    }

    // =====================================================================
    // HNSW
    // =====================================================================
    static class HNSW {
        static class Node {
            VectorItem item;
            int maxLyr;
            List<List<Integer>> nbrs;

            Node(VectorItem item, int maxLyr, List<List<Integer>> nbrs) {
                this.item = item;
                this.maxLyr = maxLyr;
                this.nbrs = nbrs;
            }
        }

        Map<Integer, Node> G = new HashMap<>();
        int M, M0, ef_build;
        float mL;
        int topLayer = -1;
        int entryPt = -1;
        Random rng = new Random(42);

        HNSW(int m, int efBuild) {
            this.M = m;
            this.M0 = 2 * m;
            this.ef_build = efBuild;
            this.mL = (float) (1.0 / Math.log(m));
        }

        int randLevel() {
            return (int) Math.floor(-Math.log(rng.nextFloat()) * mL);
        }

        List<DistResult> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            PriorityQueue<DistResult> cands = new PriorityQueue<>();
            PriorityQueue<DistResult> found = new PriorityQueue<>(Collections.reverseOrder());

            float d0 = dist.apply(q, G.get(ep).item.emb);
            vis.add(ep);
            cands.add(new DistResult(d0, ep));
            found.add(new DistResult(d0, ep));

            while (!cands.isEmpty()) {
                DistResult c = cands.poll();
                if (found.size() >= ef && c.dist > found.peek().dist)
                    break;
                if (lyr >= G.get(c.id).nbrs.size())
                    continue;
                for (int nid : G.get(c.id).nbrs.get(lyr)) {
                    if (vis.contains(nid) || !G.containsKey(nid))
                        continue;
                    vis.add(nid);
                    float nd = dist.apply(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().dist) {
                        cands.add(new DistResult(nd, nid));
                        found.add(new DistResult(nd, nid));
                        if (found.size() > ef)
                            found.poll();
                    }
                }
            }
            List<DistResult> res = new ArrayList<>();
            while (!found.isEmpty())
                res.add(found.poll());
            Collections.sort(res);
            return res;
        }

        List<Integer> selectNbrs(List<DistResult> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++)
                r.add(cands.get(i).id);
            return r;
        }

        void insert(VectorItem item, DistFn dist) {
            int id = item.id;
            int lvl = randLevel();
            List<List<Integer>> nbrs = new ArrayList<>();
            for (int i = 0; i <= lvl; i++)
                nbrs.add(new ArrayList<>());
            G.put(id, new Node(item, lvl, nbrs));

            if (entryPt == -1) {
                entryPt = id;
                topLayer = lvl;
                return;
            }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<DistResult> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty())
                        ep = W.get(0).id;
                }
            }
            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<DistResult> W = searchLayer(item.emb, ep, ef_build, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, sel);

                for (int nid : sel) {
                    if (!G.containsKey(nid))
                        continue;
                    Node nNode = G.get(nid);
                    while (nNode.nbrs.size() <= lc)
                        nNode.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nNode.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<DistResult> ds = new ArrayList<>();
                        for (int c : conn)
                            if (G.containsKey(c))
                                ds.add(new DistResult(dist.apply(nNode.item.emb, G.get(c).item.emb), c));
                        Collections.sort(ds);
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++)
                            conn.add(ds.get(i).id);
                    }
                }
                if (!W.isEmpty())
                    ep = W.get(0).id;
            }
            if (lvl > topLayer) {
                topLayer = lvl;
                entryPt = id;
            }
        }

        List<DistResult> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1)
                return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<DistResult> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty())
                        ep = W.get(0).id;
                }
            }
            List<DistResult> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k)
                W = W.subList(0, k);
            return W;
        }

        void remove(int id) {
            if (!G.containsKey(id))
                return;
            for (Node nd : G.values())
                for (List<Integer> layer : nd.nbrs)
                    layer.remove((Integer) id);
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet())
                    if (nid != id) {
                        entryPt = nid;
                        break;
                    }
            }
            G.remove(id);
        }

        static class GraphInfo {
            int topLayer, nodeCount;
            List<Integer> nodesPerLayer = new ArrayList<>(), edgesPerLayer = new ArrayList<>();

            static class NV {
                int id;
                String metadata, category;
                int maxLyr;
            }

            static class EV {
                int src, dst, lyr;
            }

            List<NV> nodes = new ArrayList<>();
            List<EV> edges = new ArrayList<>();
        }

        GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            for (int i = 0; i < maxL; i++) {
                gi.nodesPerLayer.add(0);
                gi.edgesPerLayer.add(0);
            }
            for (Map.Entry<Integer, Node> e : G.entrySet()) {
                int id = e.getKey();
                Node nd = e.getValue();
                GraphInfo.NV nv = new GraphInfo.NV();
                nv.id = id;
                nv.metadata = nd.item.metadata;
                nv.category = nd.item.category;
                nv.maxLyr = nd.maxLyr;
                gi.nodes.add(nv);
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                    if (lc < nd.nbrs.size())
                        for (int nid : nd.nbrs.get(lc))
                            if (id < nid) {
                                gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                                GraphInfo.EV ev = new GraphInfo.EV();
                                ev.src = id;
                                ev.dst = nid;
                                ev.lyr = lc;
                                gi.edges.add(ev);
                            }
                }
            }
            return gi;
        }

        int size() {
            return G.size();
        }
    }

    // =====================================================================
    // VECTOR DATABASE
    // =====================================================================
    static class VectorDB {
        Map<Integer, VectorItem> store = new HashMap<>();
        BruteForce bf = new BruteForce();
        KDTree kdt;
        HNSW hnsw;
        int nextId = 1;
        final int dims;

        VectorDB(int d) {
            this.dims = d;
            this.kdt = new KDTree(d);
            this.hnsw = new HNSW(16, 200);
        }

        synchronized int insert(String meta, String cat, float[] emb, DistFn dist) {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            TxLog.append("V", "I", v.id, meta, cat, emb);
            return v.id;
        }

        synchronized boolean remove(int id) {
            if (!store.containsKey(id))
                return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        }

        synchronized boolean removeByMetadata(String meta) {
            List<Integer> toRemove = new ArrayList<>();
            for (VectorItem v : store.values()) {
                if (v.metadata != null && (v.metadata.equals(meta) || meta.startsWith(v.metadata))) {
                    toRemove.add(v.id);
                }
            }
            boolean removedAny = false;
            for (int id : toRemove) {
                if (remove(id)) removedAny = true;
            }
            return removedAny;
        }

        synchronized void clear() {
            store.clear();
            bf = new BruteForce();
            kdt = new KDTree(dims);
            hnsw = new HNSW(16, 200);
            nextId = 1;
        }

        static class Hit {
            int id;
            String meta, cat;
            float[] emb;
            float dist;
        }

        static class SearchOut {
            List<Hit> hits = new ArrayList<>();
            long us;
            String algo, metric;
        }

        synchronized SearchOut search(float[] q, int k, String metric, String algo) {
            DistFn dfn = getDistFn(metric);
            long t0 = System.nanoTime();
            List<DistResult> raw;
            if ("bruteforce".equals(algo))
                raw = bf.knn(q, k, dfn);
            else if ("kdtree".equals(algo))
                raw = kdt.knn(q, k, dfn);
            else
                raw = hnsw.knn(q, k, 50, dfn);
            long us = (System.nanoTime() - t0) / 1000;
            SearchOut out = new SearchOut();
            out.us = us;
            out.algo = algo;
            out.metric = metric;
            for (DistResult r : raw) {
                if (store.containsKey(r.id)) {
                    VectorItem vi = store.get(r.id);
                    Hit h = new Hit();
                    h.id = vi.id;
                    h.meta = vi.metadata;
                    h.cat = vi.category;
                    h.emb = vi.emb;
                    h.dist = r.dist;
                    out.hits.add(h);
                }
            }
            return out;
        }

        static class BenchOut {
            long bfUs, kdUs, hnswUs;
            int n;
        }

        synchronized BenchOut benchmark(float[] q, int k, String metric) {
            DistFn dfn = getDistFn(metric);
            BenchOut bo = new BenchOut();
            long t0 = System.nanoTime();
            bf.knn(q, k, dfn);
            bo.bfUs = (System.nanoTime() - t0) / 1000;
            t0 = System.nanoTime();
            kdt.knn(q, k, dfn);
            bo.kdUs = (System.nanoTime() - t0) / 1000;
            t0 = System.nanoTime();
            hnsw.knn(q, k, 50, dfn);
            bo.hnswUs = (System.nanoTime() - t0) / 1000;
            bo.n = store.size();
            return bo;
        }

        synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized HNSW.GraphInfo hnswInfo() {
            return hnsw.getInfo();
        }

        synchronized int size() {
            return store.size();
        }
    }

    // =====================================================================
    // JSON HELPERS
    // =====================================================================
    static String jS(String s) {
        if (s == null)
            return "\"\"";
        StringBuilder o = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"')
                o.append("\\\"");
            else if (c == '\\')
                o.append("\\\\");
            else if (c == '\n')
                o.append("\\n");
            else if (c == '\r')
                o.append("\\r");
            else if (c == '\t')
                o.append("\\t");
            else
                o.append(c);
        }
        return o.append('"').toString();
    }

    static String jVec(float[] v) {
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0)
                ss.append(",");
            ss.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return ss.append(']').toString();
    }

    static float[] parseVec(String s) {
        if (s == null || s.isEmpty())
            return new float[0];
        String[] parts = s.replace("[", "").replace("]", "").split(",");
        List<Float> v = new ArrayList<>();
        for (String p : parts) {
            try {
                v.add(Float.parseFloat(p.trim()));
            } catch (Exception ignored) {
            }
        }
        float[] res = new float[v.size()];
        for (int i = 0; i < v.size(); i++)
            res[i] = v.get(i);
        return res;
    }

    static String extractStr(String body, String key) {
        String match = "\"" + key + "\"";
        int p = body.indexOf(match);
        if (p == -1)
            return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t'))
            p++;
        if (p >= body.length() || body.charAt(p) != '"')
            return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            if (body.charAt(p) == '"')
                break;
            if (body.charAt(p) == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"':
                        result.append('"');
                        break;
                    case '\\':
                        result.append('\\');
                        break;
                    case 'n':
                        result.append('\n');
                        break;
                    case 'r':
                        result.append('\r');
                        break;
                    case 't':
                        result.append('\t');
                        break;
                    default:
                        result.append(body.charAt(p));
                        break;
                }
            } else {
                result.append(body.charAt(p));
            }
            p++;
        }
        return result.toString();
    }

    static int extractInt(String body, String key, int def) {
        String match = "\"" + key + "\"";
        int p = body.indexOf(match);
        if (p == -1)
            return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t'))
            p++;
        int e = p;
        while (e < body.length() && Character.isDigit(body.charAt(e)))
            e++;
        try {
            return Integer.parseInt(body.substring(p, e));
        } catch (Exception ex) {
            return def;
        }
    }

    static class ParsedBody {
        String meta, cat;
        float[] emb;
    }

    static ParsedBody parseBodyObj(String b) {
        ParsedBody r = new ParsedBody();
        r.meta = extractStr(b, "metadata");
        r.cat = extractStr(b, "category");
        int p = b.indexOf("\"embedding\"");
        if (p != -1) {
            p = b.indexOf('[', p);
            if (p != -1) {
                int e = b.indexOf(']', p);
                if (e != -1)
                    r.emb = parseVec(b.substring(p + 1, e));
            }
        }
        return r;
    }

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0 || words[0].isEmpty())
            return new ArrayList<>();
        if (words.length <= chunkWords)
            return Collections.singletonList(text);
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i)
                    chunk.append(' ');
                chunk.append(words[j]);
            }
            chunks.add(chunk.toString());
            if (end == words.length)
                break;
        }
        return chunks;
    }

    // =====================================================================
    // PERSISTENCE (TX LOG)
    // =====================================================================
    static class TxLog {
        static final String FILE_V = "demo_vectors.jsonl";
        static final String FILE_D = "rag_docs.jsonl";
        static boolean isReplaying = false;

        static synchronized void append(String type, String action, int id, String s1, String s2, float[] emb) {
            if (isReplaying)
                return;
            String file = type.equals("V") ? FILE_V : FILE_D;
            try (FileWriter fw = new FileWriter(file, true)) {
                StringBuilder sb = new StringBuilder();
                sb.append("{\"type\":\"").append(type).append("\",\"action\":\"").append(action).append("\",\"id\":")
                        .append(id);
                if (s1 != null)
                    sb.append(",\"metadata\":").append(jS(s1));
                if (s2 != null)
                    sb.append(",\"category\":").append(jS(s2));
                if (emb != null)
                    sb.append(",\"embedding\":").append(jVec(emb));
                sb.append("}\n");
                fw.write(sb.toString());
            } catch (Exception e) {
            }
        }

        static synchronized void clear() {
            try {
                new FileWriter(FILE_V, false).close();
                new FileWriter(FILE_D, false).close();
            } catch (Exception e) {}
        }

        static synchronized void compact(VectorDB db, DocumentDB docDB) {
            if (isReplaying) return;
            try (FileWriter fw = new FileWriter(FILE_V, false)) {
                for (VectorItem v : db.store.values()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("{\"type\":\"V\",\"action\":\"I\",\"id\":").append(v.id);
                    if (v.metadata != null) sb.append(",\"metadata\":").append(jS(v.metadata));
                    if (v.category != null) sb.append(",\"category\":").append(jS(v.category));
                    sb.append(",\"embedding\":").append(jVec(v.emb)).append("}\n");
                    fw.write(sb.toString());
                }
            } catch (Exception e) {}
            try (FileWriter fw = new FileWriter(FILE_D, false)) {
                for (DocItem d : docDB.store.values()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("{\"type\":\"D\",\"action\":\"I\",\"id\":").append(d.id);
                    if (d.title != null) sb.append(",\"metadata\":").append(jS(d.title));
                    if (d.text != null) sb.append(",\"category\":").append(jS(d.text));
                    sb.append(",\"embedding\":").append(jVec(d.emb)).append("}\n");
                    fw.write(sb.toString());
                    if (!d.active) {
                        fw.write("{\"type\":\"D\",\"action\":\"T\",\"id\":" + d.id + "}\n");
                    }
                }
            } catch (Exception e) {}
        }

        static void replay(VectorDB db, DocumentDB docDB) {
            isReplaying = true;
            File legacy = new File("vector_db.jsonl");
            if (legacy.exists()) {
                replayFile(legacy, db, docDB);
                legacy.delete();
            } else {
                replayFile(new File(FILE_V), db, docDB);
                replayFile(new File(FILE_D), db, docDB);
            }
            
            // Rebuild indices
            db.kdt.rebuild(new ArrayList<>(db.store.values()));
            for (VectorItem v : db.store.values()) {
                db.bf.insert(v);
                db.hnsw.insert(v, Main::cosine);
            }
            for (DocItem d : docDB.store.values()) {
                VectorItem vi = new VectorItem(d.id, d.title, "doc", d.emb);
                docDB.bf.insert(vi);
                docDB.hnsw.insert(vi, Main::cosine);
            }
            isReplaying = false;
        }

        static void replayFile(File f, VectorDB db, DocumentDB docDB) {
            if (!f.exists())
                return;
            try {
                for (String line : Files.readAllLines(f.toPath())) {
                    if (line.trim().isEmpty())
                        continue;
                    String type = extractStr(line, "type");
                    String action = extractStr(line, "action");
                    int id = extractInt(line, "id", -1);
                    if (id == -1)
                        continue;
                    if ("V".equals(type)) {
                        if ("I".equals(action)) {
                            String meta = extractStr(line, "metadata");
                            String cat = extractStr(line, "category");
                            float[] emb = parseBodyObj(line).emb;
                            db.store.put(id, new VectorItem(id, meta, cat, emb));
                            db.nextId = Math.max(db.nextId, id + 1);
                        } else if ("D".equals(action))
                            db.store.remove(id);
                    } else if ("D".equals(type)) {
                        if ("I".equals(action)) {
                            String title = extractStr(line, "metadata");
                            String text = extractStr(line, "category");
                            float[] emb = parseBodyObj(line).emb;
                            docDB.store.put(id, new DocItem(id, title, text, emb));
                            docDB.nextId = Math.max(docDB.nextId, id + 1);
                            if (docDB.dims == 0)
                                docDB.dims = emb.length;
                        } else if ("D".equals(action))
                            docDB.store.remove(id);
                        else if ("T".equals(action)) {
                            if (docDB.store.containsKey(id)) {
                                DocItem d = docDB.store.get(id);
                                d.active = !d.active;
                            }
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // =====================================================================
    // ENV READER
    // =====================================================================
    static class EnvReader {
        static Map<String, String> env = new HashMap<>();
        static {
            try {
                File f = new File(".env");
                if (f.exists()) {
                    for (String line : Files.readAllLines(f.toPath())) {
                        line = line.trim();
                        if (line.isEmpty() || line.startsWith("#"))
                            continue;
                        int idx = line.indexOf('=');
                        if (idx > 0) {
                            String k = line.substring(0, idx).trim();
                            String v = line.substring(idx + 1).trim();
                            if (v.startsWith("\"") && v.endsWith("\""))
                                v = v.substring(1, v.length() - 1);
                            env.put(k, v);
                        }
                    }
                }
            } catch (Exception e) {
            }
        }

        static String get(String key) {
            if (env.containsKey(key))
                return env.get(key);
            return System.getenv(key);
        }
    }

    // =====================================================================
    // CLOUD CLIENT (GROQ + GEMINI)
    // =====================================================================
    static class CloudClient {
        String groqKey = EnvReader.get("GROQ_API_KEY");
        String geminiKey = EnvReader.get("GEMINI_API_KEY");
        String embedModel = "gemini-embedding-2 (Gemini)";
        String genModel = "openai/gpt-oss-120b (Groq)";
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();

        String esc(String s) {
            StringBuilder o = new StringBuilder();
            for (char c : s.toCharArray()) {
                if (c == '"')
                    o.append("\\\"");
                else if (c == '\\')
                    o.append("\\\\");
                else if (c == '\n')
                    o.append("\\n");
                else if (c == '\r')
                    o.append("\\r");
                else if (c == '\t')
                    o.append("\\t");
                else
                    o.append(c);
            }
            return o.toString();
        }

        boolean isAvailable() {
            return groqKey != null && !groqKey.isEmpty() && geminiKey != null && !geminiKey.isEmpty();
        }

        float[] embed(String text) {
            if (geminiKey == null || geminiKey.isEmpty()) {
                System.out.println("No GEMINI_API_KEY");
                return new float[0];
            }
            try {
                String body = "{\"model\":\"models/gemini-embedding-2\",\"content\":{\"parts\":[{\"text\":\""
                        + esc(text) + "\"}]}}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(
                                "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key="
                                        + geminiKey))
                        .timeout(Duration.ofSeconds(30))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) {
                    System.out.println("Gemini Error: " + res.body());
                    return new float[0];
                }
                String r = res.body();
                int p = r.indexOf("\"values\"");
                if (p == -1)
                    return new float[0];
                p = r.indexOf('[', p);
                if (p == -1)
                    return new float[0];
                int e = r.indexOf(']', p);
                if (e == -1)
                    return new float[0];
                float[] raw = parseVec(r.substring(p + 1, e));
                if (raw.length > DIMS) {
                    float[] trunc = new float[DIMS];
                    System.arraycopy(raw, 0, trunc, 0, DIMS);
                    return trunc;
                }
                return raw;
            } catch (Exception e) {
                e.printStackTrace();
                return new float[0];
            }
        }

        String generate(String prompt) {
            if (groqKey == null || groqKey.isEmpty())
                return "ERROR: GROQ_API_KEY env var not set.";
            try {
                String body = "{\"model\":\"" + genModel.split(" ")[0]
                        + "\",\"messages\":[{\"role\":\"user\",\"content\":\"" + esc(prompt) + "\"}]}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create("https://api.groq.com/openai/v1/chat/completions"))
                        .timeout(Duration.ofSeconds(60))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + groqKey)
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200)
                    return "ERROR: Groq API failed: " + res.body();
                String b = res.body();
                String match = "\"content\":\"";
                int p = b.indexOf(match);
                if (p == -1)
                    return "ERROR: Unexpected response from Groq.";
                p += match.length();
                StringBuilder result = new StringBuilder();
                while (p < b.length()) {
                    if (b.charAt(p) == '"' && b.charAt(p - 1) != '\\')
                        break;
                    if (b.charAt(p) == '\\' && p + 1 < b.length()) {
                        p++;
                        char n = b.charAt(p);
                        if (n == 'n')
                            result.append('\n');
                        else if (n == 'r')
                            result.append('\r');
                        else if (n == 't')
                            result.append('\t');
                        else if (n == '"')
                            result.append('"');
                        else if (n == '\\')
                            result.append('\\');
                        else
                            result.append(n);
                    } else
                        result.append(b.charAt(p));
                    p++;
                }
                return result.toString();
            } catch (Exception e) {
                return "ERROR: " + e.getMessage();
            }
        }
    }

    // =====================================================================
    // DOCUMENT DATABASE
    // =====================================================================
    static class DocItem {
        int id;
        String title, text;
        float[] emb;
        boolean active = true;

        DocItem(int id, String title, String text, float[] emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    static class DocumentDB {
        Map<Integer, DocItem> store = new HashMap<>();
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        int nextId = 1;
        int dims = 0;

        synchronized int insert(String title, String text, float[] emb) {
            if (dims == 0)
                dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, Main::cosine);
            bf.insert(vi);
            TxLog.append("D", "I", item.id, title, text, emb);
            return item.id;
        }

        static class DocHit {
            float dist;
            DocItem doc;

            DocHit(float d, DocItem doc) {
                this.dist = d;
                this.doc = doc;
            }
        }

        synchronized List<DocHit> search(float[] q, int k, float max_dist) {
            if (store.isEmpty())
                return new ArrayList<>();
            List<DistResult> raw = (store.size() < 10) ? bf.knn(q, k * 2, Main::cosine)
                    : hnsw.knn(q, k * 2, 50, Main::cosine);
            List<DocHit> out = new ArrayList<>();
            for (DistResult r : raw) {
                if (store.containsKey(r.id)) {
                    DocItem d = store.get(r.id);
                    if (d.active && r.dist <= max_dist) {
                        out.add(new DocHit(r.dist, d));
                        if (out.size() == k)
                            break;
                    }
                }
            }
            return out;
        }

        synchronized boolean toggle(int id) {
            if (!store.containsKey(id))
                return false;
            DocItem d = store.get(id);
            d.active = !d.active;
            return d.active;
        }

        synchronized boolean remove(int id) {
            if (!store.containsKey(id))
                return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        }

        synchronized void clear() {
            store.clear();
            hnsw = new HNSW(16, 200);
            bf = new BruteForce();
            nextId = 1;
            dims = 0;
        }

        synchronized List<DocItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized int getDims() {
            return dims;
        }
    }

    // =====================================================================
    // HTTP SERVER
    // =====================================================================
    static void sendCors(HttpExchange t) {
        t.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        t.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        t.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    static void sendJson(HttpExchange t, int code, String json) throws IOException {
        sendCors(t);
        t.getResponseHeaders().add("Content-Type", "application/json");
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        t.sendResponseHeaders(code, bytes.length);
        OutputStream os = t.getResponseBody();
        os.write(bytes);
        os.close();
    }

    static Map<String, String> queryToMap(String query) {
        Map<String, String> result = new HashMap<>();
        if (query != null) {
            for (String param : query.split("&")) {
                String[] entry = param.split("=");
                if (entry.length > 1)
                    result.put(entry[0], entry[1]);
                else
                    result.put(entry[0], "");
            }
        }
        return result;
    }

    public static void main(String[] args) throws Exception {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        CloudClient cloud = new CloudClient();
        TxLog.replay(db, docDB);
        TxLog.compact(db, docDB);

        boolean cloudUp = cloud.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Cloud APIs: " + (cloudUp ? "ONLINE" : "OFFLINE (Missing GROQ_API_KEY or GEMINI_API_KEY)"));
        if (cloudUp)
            System.out.println("  embed: " + cloud.embedModel + "\n  gen: " + cloud.genModel);

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);

        server.createContext("/", (HttpExchange t) -> {
            if ("OPTIONS".equals(t.getRequestMethod())) {
                sendCors(t);
                t.sendResponseHeaders(204, -1);
                return;
            }
            String path = t.getRequestURI().getPath();

            try {
                if (path.equals("/search")) {
                    String body = new String(t.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    ParsedBody pb = parseBodyObj(body);
                    float[] q = pb.emb;
                    if (q == null || q.length != DIMS) {
                        sendJson(t, 400, "{\"error\":\"need " + DIMS + "D vector\"}");
                        return;
                    }
                    int k = extractInt(body, "k", 5);
                    String metric = extractStr(body, "metric");
                    if (metric.isEmpty())
                        metric = "cosine";
                    String algo = extractStr(body, "algo");
                    if (algo.isEmpty())
                        algo = "hnsw";
                    VectorDB.SearchOut out = db.search(q, k, metric, algo);
                    StringBuilder ss = new StringBuilder("{\"results\":[");
                    for (int i = 0; i < out.hits.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        VectorDB.Hit h = out.hits.get(i);
                        ss.append("{\"id\":").append(h.id).append(",\"metadata\":").append(jS(h.meta))
                                .append(",\"category\":").append(jS(h.cat)).append(",\"distance\":")
                                .append(String.format(Locale.US, "%.6f", h.dist)).append(",\"embedding\":")
                                .append(jVec(h.emb)).append("}");
                    }
                    ss.append("],\"latencyUs\":").append(out.us).append(",\"algo\":").append(jS(out.algo))
                            .append(",\"metric\":").append(jS(out.metric)).append("}");
                    sendJson(t, 200, ss.toString());

                } else if (path.equals("/insert")) {
                    String body = new String(t.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    ParsedBody pb = parseBodyObj(body);
                    if (pb.meta == null || pb.emb == null || pb.emb.length != DIMS) {
                        sendJson(t, 400, "{\"error\":\"invalid body\"}");
                        return;
                    }
                    int id = db.insert(pb.meta, pb.cat, pb.emb, Main::cosine);
                    sendJson(t, 200, "{\"id\":" + id + "}");

                } else if (path.equals("/embed")) {
                    Map<String, String> params = queryToMap(t.getRequestURI().getQuery());
                    String text = params.get("text");
                    if (text == null || text.isEmpty()) {
                        sendJson(t, 400, "{\"error\":\"need text\"}");
                        return;
                    }
                    float[] emb = cloud.embed(text);
                    if (emb.length == 0) {
                        sendJson(t, 500, "{\"error\":\"Embedding failed\"}");
                        return;
                    }
                    sendJson(t, 200, "{\"embedding\":" + jVec(emb) + "}");

                } else if (path.equals("/items")) {
                    List<VectorItem> items = db.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < items.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        VectorItem v = items.get(i);
                        ss.append("{\"id\":").append(v.id).append(",\"metadata\":").append(jS(v.metadata))
                                .append(",\"category\":").append(jS(v.category)).append(",\"embedding\":")
                                .append(jVec(v.emb)).append("}");
                    }
                    ss.append("]");
                    sendJson(t, 200, ss.toString());

                } else if (path.equals("/benchmark")) {
                    String body = new String(t.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    ParsedBody pb = parseBodyObj(body);
                    float[] q = pb.emb;
                    if (q == null || q.length != DIMS) {
                        sendJson(t, 400, "{\"error\":\"need " + DIMS + "D vector\"}");
                        return;
                    }
                    int k = extractInt(body, "k", 5);
                    String metric = extractStr(body, "metric");
                    if (metric.isEmpty())
                        metric = "cosine";
                    VectorDB.BenchOut b = db.benchmark(q, k, metric);
                    sendJson(t, 200, "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs + ",\"hnswUs\":"
                            + b.hnswUs + ",\"itemCount\":" + b.n + "}");

                } else if (path.equals("/hnsw-info")) {
                    HNSW.GraphInfo gi = db.hnswInfo();
                    StringBuilder ss = new StringBuilder("{\"topLayer\":").append(gi.topLayer).append(",\"nodeCount\":")
                            .append(gi.nodeCount).append(",\"nodesPerLayer\":[");
                    for (int i = 0; i < gi.nodesPerLayer.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        ss.append(gi.nodesPerLayer.get(i));
                    }
                    ss.append("],\"edgesPerLayer\":[");
                    for (int i = 0; i < gi.edgesPerLayer.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        ss.append(gi.edgesPerLayer.get(i));
                    }
                    ss.append("],\"nodes\":[");
                    for (int i = 0; i < gi.nodes.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        HNSW.GraphInfo.NV n = gi.nodes.get(i);
                        ss.append("{\"id\":").append(n.id).append(",\"metadata\":").append(jS(n.metadata))
                                .append(",\"category\":").append(jS(n.category)).append(",\"maxLyr\":").append(n.maxLyr)
                                .append("}");
                    }
                    ss.append("],\"edges\":[");
                    for (int i = 0; i < gi.edges.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        HNSW.GraphInfo.EV e = gi.edges.get(i);
                        ss.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst).append(",\"lyr\":")
                                .append(e.lyr).append("}");
                    }
                    ss.append("]}");
                    sendJson(t, 200, ss.toString());

                } else if (path.equals("/doc/insert")) {
                    String body = new String(t.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String title = extractStr(body, "title");
                    String text = extractStr(body, "text");
                    if (title.isEmpty() || text.isEmpty()) {
                        sendJson(t, 400, "{\"error\":\"need title and text\"}");
                        return;
                    }
                    List<String> chunks = chunkText(text, 250, 30);
                    List<Integer> ids = new ArrayList<>();
                    float[] firstEmb = null;
                    for (int i = 0; i < chunks.size(); i++) {
                        float[] emb = cloud.embed(chunks.get(i));
                        if (emb.length == 0) {
                            sendJson(t, 500, "{\"error\":\"Cloud API unavailable.\"}");
                            return;
                        }
                        if (firstEmb == null)
                            firstEmb = emb;
                        String chunkTitle = (chunks.size() > 1) ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                                : title;
                        ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
                    }
                    StringBuilder ss = new StringBuilder("{\"ids\":[");
                    for (int i = 0; i < ids.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        ss.append(ids.get(i));
                    }
                    ss.append("],\"chunks\":").append(chunks.size()).append(",\"dims\":").append(docDB.getDims())
                            .append(",\"embedding\":").append(firstEmb != null ? jVec(firstEmb) : "[]")
                            .append("}");
                    sendJson(t, 200, ss.toString());

                } else if (path.startsWith("/doc/toggle/")) {
                    String[] ids = path.substring("/doc/toggle/".length()).split(",");
                    boolean st = false;
                    for (String idStr : ids) {
                        st = docDB.toggle(Integer.parseInt(idStr));
                    }
                    TxLog.compact(db, docDB);
                    sendJson(t, 200, "{\"active\":" + st + "}");

                } else if (path.startsWith("/doc/delete/")) {
                    String[] ids = path.substring("/doc/delete/".length()).split(",");
                    boolean st = false;
                    for (String idStr : ids) {
                        int id = Integer.parseInt(idStr);
                        DocItem doc = docDB.store.get(id);
                        if (doc != null) {
                            String title = doc.title;
                            if (title.contains(" [")) {
                                title = title.substring(0, title.lastIndexOf(" ["));
                            }
                            db.removeByMetadata(title);
                            st = docDB.remove(id);
                        }
                    }
                    TxLog.compact(db, docDB);
                    sendJson(t, 200, "{\"success\":" + st + "}");

                } else if (path.equals("/doc/clear")) {
                    TxLog.clear();
                    db.clear();
                    docDB.clear();
                    sendJson(t, 200, "{\"success\":true}");

                } else if (path.equals("/doc/list")) {
                    List<DocItem> docs = docDB.all();
                    Map<String, List<DocItem>> grouped = new LinkedHashMap<>();
                    for (DocItem d : docs) {
                        String title = d.title;
                        if (title.contains(" [")) {
                            title = title.substring(0, title.lastIndexOf(" ["));
                        }
                        grouped.computeIfAbsent(title, k -> new ArrayList<>()).add(d);
                    }
                    StringBuilder ss = new StringBuilder("[");
                    boolean first = true;
                    for (Map.Entry<String, List<DocItem>> entry : grouped.entrySet()) {
                        if (!first) ss.append(",");
                        first = false;
                        String title = entry.getKey();
                        List<DocItem> chunks = entry.getValue();
                        StringBuilder fullText = new StringBuilder();
                        boolean allActive = true;
                        StringBuilder ids = new StringBuilder();
                        for (int j = 0; j < chunks.size(); j++) {
                            DocItem c = chunks.get(j);
                            fullText.append(c.text).append(" ");
                            if (!c.active) allActive = false;
                            if (j > 0) ids.append(",");
                            ids.append(c.id);
                        }
                        String textPreview = fullText.toString();
                        String preview = textPreview.substring(0, Math.min(textPreview.length(), 120));
                        if (textPreview.length() > 120) preview += "…";
                        int words = textPreview.split("\\s+").length;
                        ss.append("{\"id\":").append(jS(ids.toString())).append(",\"title\":").append(jS(title))
                                .append(",\"preview\":").append(jS(preview)).append(",\"words\":").append(words)
                                .append(",\"active\":").append(allActive).append("}");
                    }
                    ss.append("]");
                    sendJson(t, 200, ss.toString());

                } else if (path.equals("/doc/search")) {
                    String body = new String(t.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) {
                        sendJson(t, 400, "{\"error\":\"need question\"}");
                        return;
                    }
                    float[] qEmb = cloud.embed(question);
                    if (qEmb.length == 0) {
                        sendJson(t, 500, "{\"error\":\"Cloud API unavailable\"}");
                        return;
                    }
                    List<DocumentDB.DocHit> hits = docDB.search(qEmb, k, 0.7f);
                    StringBuilder ss = new StringBuilder("{\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        ss.append("{\"id\":").append(hits.get(i).doc.id).append(",\"title\":")
                                .append(jS(hits.get(i).doc.title))
                                .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hits.get(i).dist))
                                .append("}");
                    }
                    ss.append("]}");
                    sendJson(t, 200, ss.toString());

                } else if (path.equals("/doc/ask")) {
                    String body = new String(t.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) {
                        sendJson(t, 400, "{\"error\":\"need question\"}");
                        return;
                    }
                    float[] qEmb = cloud.embed(question);
                    if (qEmb.length == 0) {
                        sendJson(t, 500, "{\"error\":\"Cloud API unavailable\"}");
                        return;
                    }
                    List<DocumentDB.DocHit> hits = docDB.search(qEmb, k, 0.7f);
                    StringBuilder ctx = new StringBuilder();
                    for (int i = 0; i < hits.size(); i++) {
                        ctx.append("[").append(i + 1).append("] ").append(hits.get(i).doc.title).append(":\n")
                                .append(hits.get(i).doc.text).append("\n\n");
                    }
                    String prompt = "You are a strict technical assistant. You must ONLY answer the user's question using information from the provided context. If the provided context does not contain the answer, you must reply exactly with: 'I do not know because it is not in my database.' Do NOT use outside general knowledge.\n\nContext:\n"
                            + ctx.toString() + "Question: " + question + "\n\nAnswer:";
                    String answer = cloud.generate(prompt);
                    StringBuilder ss = new StringBuilder("{\"answer\":").append(jS(answer)).append(",\"model\":")
                            .append(jS(cloud.genModel)).append(",\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0)
                            ss.append(",");
                        ss.append("{\"id\":").append(hits.get(i).doc.id).append(",\"title\":")
                                .append(jS(hits.get(i).doc.title))
                                .append(",\"text\":").append(jS(hits.get(i).doc.text)).append(",\"distance\":")
                                .append(String.format(Locale.US, "%.4f", hits.get(i).dist)).append("}");
                    }
                    ss.append("],\"docCount\":").append(docDB.size()).append("}");
                    sendJson(t, 200, ss.toString());

                } else if (path.equals("/status")) {
                    boolean up = cloud.isAvailable();
                    String ss = "{\"ollamaAvailable\":" + up + ",\"embedModel\":" + jS(cloud.embedModel)
                            + ",\"genModel\":" + jS(cloud.genModel) + ",\"docCount\":" + docDB.size() + ",\"docDims\":"
                            + docDB.getDims() + ",\"demoDims\":" + DIMS + ",\"demoCount\":" + db.size() + "}";
                    sendJson(t, 200, ss);

                } else if (path.equals("/stats")) {
                    String ss = "{\"count\":" + db.size() + ",\"dims\":" + DIMS
                            + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
                    sendJson(t, 200, ss);

                } else if (path.equals("/") || path.equals("/index.html")) {
                    File f = new File("index.html");
                    if (!f.exists()) {
                        t.sendResponseHeaders(404, -1);
                        return;
                    }
                    byte[] bytes = Files.readAllBytes(f.toPath());
                    t.getResponseHeaders().add("Content-Type", "text/html");
                    t.sendResponseHeaders(200, bytes.length);
                    OutputStream os = t.getResponseBody();
                    os.write(bytes);
                    os.close();
                } else {
                    t.sendResponseHeaders(404, -1);
                }
            } catch (Exception e) {
                e.printStackTrace();
                t.sendResponseHeaders(500, -1);
            }
        });

        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
    }
}
