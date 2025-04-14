import heapq
from collections import defaultdict, deque


# a general class for a directed graph
class Graph:
    """
    A directed weighted graph with support for multiple graph algorithms
    """

    def __init__(self, num_of_vertex):
        self.V = num_of_vertex
        self.graph = defaultdict(list)
        self.weights = {}
        self.positive_w = True

    """
    basic actions
    """

    # add edge into the graph
    def add_edge(self, s, d, w=1):
        """
          Add a directed edge from s to d with weight w.
          Time Complexity: O(1)
          """
        self.graph[s].append(d)
        self.weights[(s, d)] = w

        if w < 0 and self.positive_w:
            self.positive_w = False

    # return all the edges of the graph
    def get_edges(self):
        """
        Get all edges in the graph as a list of tuples (u, v).

        Time Complexity: O(E)
        """
        return [(u, v) for u in self.graph for v in self.graph[u]]

    def in_degrees(self):
        """
        Compute the in-degree of each vertex.

        Time Complexity: O(E)
        """
        in_deg = [0] * self.V
        for u in self.graph:
            for v in self.graph[u]:
                in_deg[v] += 1
        return in_deg

    def out_degrees(self):
        """
        Compute the out-degree of each vertex.

        Time Complexity: O(V)
        """
        return [len(self.graph[i]) for i in range(self.V)]

    # transpose the edge matrix
    def transpose(self):
        """
        Return the transpose of the graph (all edges reversed).
        Time Complexity: O(V + E)
        """
        g = Graph(self.V)
        for u in self.graph:
            for v in self.graph[u]:
                g.add_edge(v, u, self.weights[(u, v)])
        return g

    def find(self, parent, i):
        """
        Find the representative of the set containing i.
        Uses path compression.
        Time Complexity: O(a(N)), where a is the inverse Ackermann function.
        """
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def union(self, parent, rank, x, y):
        """
        Union two sets x and y using rank.
        Time Complexity: O(a(N))
        """
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    """
    graph algorithms
    """

    #### ds
    def is_eulerian(graph, V):
        """
        Check if a directed graph has an Eulerian path or circuit.

        Args:
            graph (dict[int, list[int]]): adjacency list
            V (int): number of vertices

        Returns:
            (bool, list[int]): True and path if Eulerian path or circuit exists, else False and empty list
        Time Complexity: O(V + E)
        """
        in_deg = [0] * V
        out_deg = [0] * V
        temp_graph = defaultdict(deque)

        for u in graph:
            for v in graph[u]:
                temp_graph[u].append(v)
                out_deg[u] += 1
                in_deg[v] += 1

        start_nodes = end_nodes = 0
        start = 0

        for i in range(V):
            if out_deg[i] - in_deg[i] == 1:
                start_nodes += 1
                start = i
            elif in_deg[i] - out_deg[i] == 1:
                end_nodes += 1
            elif in_deg[i] != out_deg[i]:
                return False, []

        if not (start_nodes == 0 and end_nodes == 0 or start_nodes == 1 and end_nodes == 1):
            return False, []

        if start_nodes == 0 and end_nodes == 0:
            for i in range(V):
                if out_deg[i]:
                    start = i
                    break

        stack = [start]
        path = []
        local_graph = {u: deque(vs) for u, vs in temp_graph.items()}

        while stack:
            u = stack[-1]
            if local_graph.get(u):
                stack.append(local_graph[u].popleft())
            else:
                path.append(stack.pop())
        path = path[::-1]

        total_edges = sum(len(v) for v in graph.values())
        if len(path) == total_edges + 1:
            return True, path
        else:
            return False, []

    def has_hamiltonian_path(self):
        """
        Check if the graph has a Hamiltonian path
        Returns (True, path) if it exists, else (False, []).
        Time Complexity: O(V!)
        """

        def backtrack(v, visited, path):
            if len(path) == self.V:
                return True
            for u in self.graph[v]:
                if not visited[u]:
                    visited[u] = True
                    path.append(u)
                    if backtrack(u, visited, path):
                        return True
                    path.pop()
                    visited[u] = False
            return False

        for start in range(self.V):
            visited = [False] * self.V
            path = [start]
            visited[start] = True
            if backtrack(start, visited, path):
                return True, path
        return False, []

    def has_hamiltonian_circuit(self):
        """
        Check if the graph has a Hamiltonian circuit
        Returns (True, cycle) if it exists, else (False, []).
        Time Complexity: O(V!)
        """

        def backtrack(v, visited, path, start):
            if len(path) == self.V:
                if start in self.graph[v]:
                    path.append(start)
                    return True
                return False
            for u in self.graph[v]:
                if not visited[u]:
                    visited[u] = True
                    path.append(u)
                    if backtrack(u, visited, path, start):
                        return True
                    path.pop()
                    visited[u] = False
            return False

        for start in range(self.V):
            visited = [False] * self.V
            path = [start]
            visited[start] = True
            if backtrack(start, visited, path, start):
                return True, path
        return False, []

    #### algo 1

    # the recursive dfs inner func
    def __dfs(self, u, visited, pre_order, post_order, print_forest):
        """
        Internal DFS utility with preorder and postorder recording.
        Time Complexity: O(V + E)
        """
        visited[u] = True
        pre_order.append(u)
        for v in self.graph[u]:
            if not visited[v]:
                if print_forest:
                    print(f"{u} -> {v}")
                self.__dfs(v, visited, pre_order, post_order, print_forest)
        post_order.append(u)

    # returns the vertex in the dfs order, if wanted prints the dfs forest
    def dfs(self, start=0, print_forest=True):
        """
        Perform DFS traversal from a given node.
        Returns pre-order and post-order lists.
        Time Complexity: O(V + E)
        """
        visited = [False] * self.V
        pre_order = []
        post_order = []
        self.__dfs(start, visited, pre_order, post_order, print_forest)
        return pre_order, post_order

    # returns the vertex in the bfs order, if wanted prints the bfs tree
    def bfs(self, s=0, print_tree=True):
        """
        Perform BFS traversal from node s.
        Returns the BFS order of vertices.
        Time Complexity: O(V + E)
        """
        visited = [False] * self.V
        queue = deque([s])
        visited[s] = True
        order = [s]

        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    order.append(v)
                    if print_tree:
                        print(f"{u} -> {v}")
        return order

    # returns the scc, if wanted, prints the SCC - O(V+E)
    def scc(self, print_scc=True):
        """
        Compute and return the strongly connected components using Kosaraju's algorithm.
        Time Complexity: O(V + E)
        """
        _, post_order = self.dfs(print_forest=False)
        g_trans = self.transpose()
        visited = [False] * self.V
        scc_list = []

        while post_order:
            u = post_order.pop()
            if not visited[u]:
                pre, _ = [], []
                g_trans.__dfs(u, visited, pre, _, print_scc)
                if print_scc:
                    print("/")
                scc_list.append(pre)
        return scc_list

    # returns the MST calced by Prim's algorithm, if wanted, prints it
    def mst_Prim(self, to_print=True):
        """
        Compute the Minimum Spanning Tree using Prim's algorithm.
        Returns the list of edges and total weight.
        Time Complexity: O((V + E) log V)
        """
        visited = [False] * self.V
        min_heap = [(0, 0)]  # (weight, vertex)
        mst_weight = 0
        mst_edges = []

        while min_heap:
            weight, u = heapq.heappop(min_heap)
            if visited[u]:
                continue
            visited[u] = True
            mst_weight += weight
            for i, v in enumerate(self.graph[u]):
                w = self.weights[u, v][0]
                if not visited[v]:
                    heapq.heappush(min_heap, (w, v))
                    mst_edges.append((u, v, w))
        if to_print:
            print(f"Prim's MST total weight: {mst_weight}")
            for edge in mst_edges:
                print(edge)
        return mst_edges, mst_weight

    # returns the MST calced by Kruskal's algorithm, if wanted, prints it
    def mst_Kruskal(self, to_print=True):
        """
        Compute the MST using Kruskal’s algorithm.
        Returns the list of edges and total weight.
        Time Complexity: O(E log E)
        """
        edges = []
        for u in self.graph:
            for i, v in enumerate(self.graph[u]):
                edges.append((self.weights[u, v][i], u, v))
        edges.sort()

        parent = list(range(self.V))
        rank = [0] * self.V
        mst_weight = 0
        mst_edges = []

        for w, u, v in edges:
            if self.find(parent, u) != self.find(parent, v):
                self.union(parent, rank, u, v)
                mst_edges.append((u, v, w))
                mst_weight += w

        if to_print:
            print("Kruskal's MST total weight:", mst_weight)
            for edge in mst_edges:
                print(edge)

        return mst_edges, mst_weight

    # returns the shortest path from u to v and its weight calced using Dijkstra, if wanted, prints it
    def shortest_path_Dijkstra(self, u, v, to_print=True):
        """
        Compute shortest path from u to v using Dijkstra’s algorithm.
        Time Complexity: O((V + E) log V)
        """
        dist = [float('inf')] * self.V
        prev = [None] * self.V
        dist[u] = 0
        heap = [(0, u)]

        while heap:
            d, node = heapq.heappop(heap)
            if node == v:
                break
            for i, neighbor in enumerate(self.graph[node]):
                weight = self.weights[node, neighbor][0]
                if dist[node] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node] + weight
                    prev[neighbor] = node
                    heapq.heappush(heap, (dist[neighbor], neighbor))

        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if to_print:
            print("Dijkstra's path:", path)

        return path, dist[v]

    # returns the shortest path from u to v and its weight calced by Bellman Ford, if wanted, prints it
    def shortest_path_Bellman_Ford(self, u, v, to_print=True):
        """
        Compute shortest path from u to v using Bellman-Ford.
        Detects negative weight cycles.
        Time Complexity: O(V * E)
        """
        dist = [float('inf')] * self.V
        prev = [None] * self.V
        dist[u] = 0

        for _ in range(self.V - 1):
            for src in self.graph:
                for i, dest in enumerate(self.graph[src]):
                    weight = self.weights[src, dest][i]
                    if dist[src] + weight < dist[dest]:
                        dist[dest] = dist[src] + weight
                        prev[dest] = src

        # check for negative-weight cycles
        for src in self.graph:
            for i, dest in enumerate(self.graph[src]):
                weight = self.weights[src, dest][i]
                if dist[src] + weight < dist[dest]:
                    raise ValueError("Graph contains a negative weight cycle")

        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if to_print:
            print("Bellman-Ford path:", path)
        return path, dist[v]

    # returns a matrix of all pairs shortest paths calced by Johnson algorithm
    def apsp(self):
        """
        All-pairs shortest paths using Johnson's algorithm.
        Combines Bellman-Ford and Dijkstra.
        Time Complexity: O(V * E + V * (V + E) log V)
        """
        # add new vertex and connect it to all vertices with edge 0
        temp_graph = Graph(self.V + 1)
        for u in self.graph:
            for i, v in enumerate(self.graph[u]):
                w = self.weights[u, v][i]
                temp_graph.add_edge(u, v, w)
        for i in range(self.V):
            temp_graph.add_edge(self.V, i, 0)

        # run Bellman-Ford from new vertex to find h(v)
        h = [float('inf')] * (self.V + 1)
        h[self.V] = 0
        for _ in range(self.V):
            for u in temp_graph.graph:
                for i, v in enumerate(temp_graph.graph[u]):
                    w = temp_graph.weights[u, v][i]
                    if h[u] + w < h[v]:
                        h[v] = h[u] + w

        # reweight original edges
        new_weights = defaultdict(list)
        for u in self.graph:
            for i, v in enumerate(self.graph[u]):
                w = self.weights[u, v][i]
                new_weights[u, v].append(w + h[u] - h[v])

        # run Dijkstra from each vertex
        apsp_matrix = [[float('inf')] * self.V for _ in range(self.V)]
        for src in range(self.V):
            dist = [float('inf')] * self.V
            dist[src] = 0
            heap = [(0, src)]

            while heap:
                d, u = heapq.heappop(heap)
                for i, v in enumerate(self.graph[u]):
                    w = new_weights[u, v][i]
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        heapq.heappush(heap, (dist[v], v))

            for v in range(self.V):
                if dist[v] < float('inf'):
                    apsp_matrix[src][v] = dist[v] - h[src] + h[v]

        return apsp_matrix
