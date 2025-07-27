import tkinter as tk
from tkinter import simpledialog, messagebox, ttk, colorchooser
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from collections import defaultdict

class PathfindingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Pathfinding Visualizer")
        self.root.geometry("1200x800")
        
        # Graph and visualization settings
        self.G = nx.Graph()
        self.pos = {}
        self.node_colors = defaultdict(lambda: '#87CEEB')
        self.edge_colors = defaultdict(lambda: '#808080')
        self.highlighted_path = []
        self.node_positions_fixed = False
        
        # Create UI
        self.setup_ui()
        self.draw_graph()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Graph controls
        ttk.Label(control_frame, text="Graph Operations", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        ttk.Button(control_frame, text="Add Node", command=self.add_node).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Add Edge", command=self.add_edge).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Remove Node", command=self.remove_node).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Remove Edge", command=self.remove_edge).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Algorithm selection
        ttk.Label(control_frame, text="Algorithm", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        self.algorithm_choice = tk.StringVar()
        self.algorithm_choice.set("Dijkstra")
        algorithm_menu = ttk.Combobox(control_frame, textvariable=self.algorithm_choice, state='readonly', width=20)
        algorithm_menu['values'] = ('Dijkstra', 'A*', 'Floyd-Warshall', 'Bellman-Ford', 'Bidirectional Dijkstra', 'BFS', 'DFS')
        algorithm_menu.pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Pathfinding controls
        ttk.Label(control_frame, text="Pathfinding", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        ttk.Button(control_frame, text="Find Path", command=self.find_path).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Find All Paths from Node", command=self.find_all_paths).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Highlights", command=self.clear_highlights).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Graph presets
        ttk.Label(control_frame, text="Graph Presets", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        ttk.Button(control_frame, text="Grid Graph", command=self.create_grid_graph).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Random Graph", command=self.create_random_graph).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Complete Graph", command=self.create_complete_graph).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clear Graph", command=self.clear_graph).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Visualization options
        ttk.Label(control_frame, text="Visualization", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        
        self.show_weights = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Weights", variable=self.show_weights, 
                       command=self.draw_graph).pack(fill=tk.X, pady=2)
        
        self.directed = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Directed Graph", variable=self.directed, 
                       command=self.toggle_directed).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Relayout Graph", command=self.relayout_graph).pack(fill=tk.X, pady=2)
        
        # Right panel for graph
        graph_frame = ttk.LabelFrame(main_frame, text="Graph Visualization", padding=10)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        # Bind mouse events for interactive node placement
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def draw_graph(self):
        self.ax.clear()
        
        if not self.G.nodes():
            self.ax.text(0.5, 0.5, 'Empty Graph\nAdd nodes to begin', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=14, color='gray')
            self.canvas.draw_idle()
            return
            
        # Update positions if needed
        if not self.node_positions_fixed or len(self.pos) != len(self.G.nodes()):
            self.pos = nx.spring_layout(self.G, pos=self.pos if self.pos else None, 
                                       iterations=50, k=2/np.sqrt(len(self.G.nodes())))
        
        # Prepare colors
        node_colors = [self.node_colors[node] for node in self.G.nodes()]
        
        # Draw graph
        if self.directed.get():
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_colors, 
                                 node_size=700, alpha=0.9)
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=12, font_weight='bold')
            
            # Draw edges with proper coloring for directed graph
            for edge in self.G.edges():
                color = self.edge_colors[edge] if edge in self.highlighted_path else '#808080'
                nx.draw_networkx_edges(self.G, self.pos, [(edge[0], edge[1])], ax=self.ax,
                                     edge_color=color, width=2, alpha=0.7, arrows=True,
                                     arrowsize=20, arrowstyle='->')
        else:
            nx.draw(self.G, self.pos, ax=self.ax, with_labels=True, node_color=node_colors,
                   edge_color=[self.edge_colors[edge] for edge in self.G.edges()],
                   node_size=700, font_size=12, font_weight='bold', width=2, alpha=0.9)
        
        # Draw edge weights if enabled
        if self.show_weights.get() and nx.get_edge_attributes(self.G, 'weight'):
            edge_labels = nx.get_edge_attributes(self.G, 'weight')
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, ax=self.ax, font_size=10)
        
        # Add title with graph info
        self.ax.set_title(f"Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()}", 
                         fontsize=12, pad=20)
        
        self.canvas.draw_idle()
        
    def on_click(self, event):
        """Handle mouse clicks for interactive node placement"""
        if event.inaxes != self.ax:
            return
            
        # Convert click coordinates to data coordinates
        x, y = event.xdata, event.ydata
        
        # Find if click is near any node
        for node, (nx, ny) in self.pos.items():
            if np.sqrt((x - nx)**2 + (y - ny)**2) < 0.05:
                self.status_var.set(f"Selected node: {node}")
                return
                
    def add_node(self):
        node = simpledialog.askstring("Add Node", "Enter node name:")
        if node and node not in self.G:
            self.G.add_node(node)
            self.node_positions_fixed = False
            self.draw_graph()
            self.status_var.set(f"Added node: {node}")
        elif node in self.G:
            messagebox.showwarning("Warning", f"Node '{node}' already exists")
            
    def add_edge(self):
        if len(self.G.nodes()) < 2:
            messagebox.showwarning("Warning", "Need at least 2 nodes to add an edge")
            return
            
        node1 = simpledialog.askstring("Add Edge", "Enter first node:")
        if not node1 or node1 not in self.G:
            messagebox.showerror("Error", f"Node '{node1}' not found")
            return
            
        node2 = simpledialog.askstring("Add Edge", "Enter second node:")
        if not node2 or node2 not in self.G:
            messagebox.showerror("Error", f"Node '{node2}' not found")
            return
            
        weight = simpledialog.askstring("Edge Weight", "Enter edge weight (default: 1):")
        weight = float(weight) if weight else 1.0
        
        self.G.add_edge(node1, node2, weight=weight)
        self.draw_graph()
        self.status_var.set(f"Added edge: {node1} - {node2} (weight: {weight})")
        
    def remove_node(self):
        if not self.G.nodes():
            messagebox.showwarning("Warning", "No nodes to remove")
            return
            
        node = simpledialog.askstring("Remove Node", "Enter node name:")
        if node and node in self.G:
            self.G.remove_node(node)
            if node in self.pos:
                del self.pos[node]
            self.draw_graph()
            self.status_var.set(f"Removed node: {node}")
        else:
            messagebox.showerror("Error", f"Node '{node}' not found")
            
    def remove_edge(self):
        if not self.G.edges():
            messagebox.showwarning("Warning", "No edges to remove")
            return
            
        node1 = simpledialog.askstring("Remove Edge", "Enter first node:")
        node2 = simpledialog.askstring("Remove Edge", "Enter second node:")
        
        if self.G.has_edge(node1, node2):
            self.G.remove_edge(node1, node2)
            self.draw_graph()
            self.status_var.set(f"Removed edge: {node1} - {node2}")
        else:
            messagebox.showerror("Error", f"Edge between '{node1}' and '{node2}' not found")
            
    def toggle_directed(self):
        if self.directed.get():
            self.G = self.G.to_directed()
        else:
            self.G = self.G.to_undirected()
        self.draw_graph()
        
    def relayout_graph(self):
        self.node_positions_fixed = False
        self.pos = {}
        self.draw_graph()
        
    def clear_highlights(self):
        self.node_colors.clear()
        self.edge_colors.clear()
        self.highlighted_path = []
        for node in self.G.nodes():
            self.node_colors[node] = '#87CEEB'
        for edge in self.G.edges():
            self.edge_colors[edge] = '#808080'
        self.draw_graph()
        self.status_var.set("Highlights cleared")
        
    def heuristic(self, a, b):
        """Heuristic function for A* algorithm"""
        if a in self.pos and b in self.pos:
            return np.sqrt((self.pos[a][0] - self.pos[b][0])**2 + 
                          (self.pos[a][1] - self.pos[b][1])**2)
        return 0
        
    def find_path(self):
        if len(self.G.nodes()) < 2:
            messagebox.showwarning("Warning", "Need at least 2 nodes")
            return
            
        start = simpledialog.askstring("Find Path", "Enter start node:")
        if not start or start not in self.G:
            messagebox.showerror("Error", f"Node '{start}' not found")
            return
            
        end = simpledialog.askstring("Find Path", "Enter end node:")
        if not end or end not in self.G:
            messagebox.showerror("Error", f"Node '{end}' not found")
            return
            
        algorithm = self.algorithm_choice.get()
        
        try:
            if algorithm == 'Dijkstra':
                path = nx.dijkstra_path(self.G, start, end)
                cost = nx.dijkstra_path_length(self.G, start, end)
            elif algorithm == 'A*':
                path = nx.astar_path(self.G, start, end, heuristic=self.heuristic)
                cost = sum(self.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            elif algorithm == 'Bellman-Ford':
                path = nx.bellman_ford_path(self.G, start, end)
                cost = sum(self.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            elif algorithm == 'Bidirectional Dijkstra':
                cost, path = nx.bidirectional_dijkstra(self.G, start, end)
            elif algorithm == 'BFS':
                path = nx.shortest_path(self.G, start, end)
                cost = len(path) - 1
            elif algorithm == 'DFS':
                # Custom DFS implementation
                visited = set()
                path = []
                if self._dfs(start, end, visited, path):
                    cost = sum(self.G[path[i]][path[i+1]].get('weight', 1) for i in range(len(path)-1))
                else:
                    raise nx.NetworkXNoPath(f"No path between {start} and {end}")
            else:
                messagebox.showerror("Error", "Floyd-Warshall requires 'Find All Paths' option")
                return
                
            # Highlight path
            self.clear_highlights()
            self.highlighted_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # Color nodes
            for node in path:
                if node == start:
                    self.node_colors[node] = '#90EE90'  # Light green
                elif node == end:
                    self.node_colors[node] = '#FFB6C1'  # Light pink
                else:
                    self.node_colors[node] = '#FFD700'  # Gold
                    
            # Color edges
            for i in range(len(path)-1):
                edge = (path[i], path[i+1])
                self.edge_colors[edge] = '#FF0000'  # Red
                if not self.directed.get():
                    self.edge_colors[(edge[1], edge[0])] = '#FF0000'
                    
            self.draw_graph()
            
            path_str = ' → '.join(path)
            messagebox.showinfo(f"Path Found ({algorithm})", 
                              f"Path: {path_str}\nCost: {cost:.2f}")
            self.status_var.set(f"Path found: {start} to {end}, Cost: {cost:.2f}")
            
        except nx.NetworkXNoPath:
            messagebox.showerror("Error", f"No path between '{start}' and '{end}'")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def _dfs(self, current, end, visited, path):
        """Helper function for DFS pathfinding"""
        visited.add(current)
        path.append(current)
        
        if current == end:
            return True
            
        for neighbor in self.G.neighbors(current):
            if neighbor not in visited:
                if self._dfs(neighbor, end, visited, path):
                    return True
                    
        path.pop()
        return False
        
    def find_all_paths(self):
        if not self.G.nodes():
            messagebox.showwarning("Warning", "Graph is empty")
            return
            
        start = simpledialog.askstring("Find All Paths", "Enter start node:")
        if not start or start not in self.G:
            messagebox.showerror("Error", f"Node '{start}' not found")
            return
            
        algorithm = self.algorithm_choice.get()
        
        try:
            paths = {}
            costs = {}
            
            if algorithm == 'Floyd-Warshall':
                # Special handling for Floyd-Warshall
                pred, dist = nx.floyd_warshall_predecessor_and_distance(self.G)
                for node in self.G:
                    if node != start and start in dist and node in dist[start]:
                        path = nx.reconstruct_path(start, node, pred)
                        paths[node] = path
                        costs[node] = dist[start][node]
            else:
                # For other algorithms
                for node in self.G:
                    if node != start:
                        try:
                            if algorithm == 'Dijkstra':
                                path = nx.dijkstra_path(self.G, start, node)
                                cost = nx.dijkstra_path_length(self.G, start, node)
                            elif algorithm == 'A*':
                                path = nx.astar_path(self.G, start, node, heuristic=self.heuristic)
                                cost = sum(self.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                            elif algorithm == 'Bellman-Ford':
                                path = nx.bellman_ford_path(self.G, start, node)
                                cost = sum(self.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                            elif algorithm == 'BFS':
                                path = nx.shortest_path(self.G, start, node)
                                cost = len(path) - 1
                            else:
                                continue
                                
                            paths[node] = path
                            costs[node] = cost
                        except nx.NetworkXNoPath:
                            continue
                            
            if not paths:
                messagebox.showinfo("No Paths", f"No reachable nodes from '{start}'")
                return
                
            # Create detailed result window
            result_window = tk.Toplevel(self.root)
            result_window.title(f"All Paths from {start} ({algorithm})")
            result_window.geometry("500x400")
            
            # Create text widget with scrollbar
            text_frame = ttk.Frame(result_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=text_widget.yview)
            
            # Add results
            text_widget.insert(tk.END, f"Paths from {start} using {algorithm}:\n\n")
            
            # Sort by cost
            sorted_nodes = sorted(paths.keys(), key=lambda x: costs[x])
            
            for node in sorted_nodes:
                path_str = ' → '.join(paths[node])
                text_widget.insert(tk.END, f"To {node}: {path_str}\n")
                text_widget.insert(tk.END, f"   Cost: {costs[node]:.2f}\n\n")
                
            text_widget.config(state=tk.DISABLED)
            
            self.status_var.set(f"Found {len(paths)} paths from {start}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def create_grid_graph(self):
        rows = simpledialog.askinteger("Grid Graph", "Enter number of rows:", minvalue=2, maxvalue=10)
        if not rows:
            return
            
        cols = simpledialog.askinteger("Grid Graph", "Enter number of columns:", minvalue=2, maxvalue=10)
        if not cols:
            return
            
        self.clear_graph()
        
        # Create grid
        for i in range(rows):
            for j in range(cols):
                node = f"({i},{j})"
                self.G.add_node(node)
                
                # Add edges
                if i > 0:
                    self.G.add_edge(node, f"({i-1},{j})", weight=1)
                if j > 0:
                    self.G.add_edge(node, f"({i},{j-1})", weight=1)
                    
        self.node_positions_fixed = False
        self.draw_graph()
        self.status_var.set(f"Created {rows}x{cols} grid graph")
        
    def create_random_graph(self):
        nodes = simpledialog.askinteger("Random Graph", "Enter number of nodes:", minvalue=3, maxvalue=20)
        if not nodes:
            return
            
        prob = simpledialog.askfloat("Random Graph", "Enter edge probability (0-1):", minvalue=0, maxvalue=1)
        if prob is None:
            return
            
        self.clear_graph()
        
        # Create random graph
        self.G = nx.erdos_renyi_graph(nodes, prob)
        
        # Add random weights
        for edge in self.G.edges():
            self.G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10)
            
        # Rename nodes
        mapping = {i: str(i) for i in range(nodes)}
        self.G = nx.relabel_nodes(self.G, mapping)
        
        self.node_positions_fixed = False
        self.draw_graph()
        self.status_var.set(f"Created random graph with {nodes} nodes")
        
    def create_complete_graph(self):
        nodes = simpledialog.askinteger("Complete Graph", "Enter number of nodes:", minvalue=2, maxvalue=10)
        if not nodes:
            return
            
        self.clear_graph()
        
        # Create complete graph
        self.G = nx.complete_graph(nodes)
        
        # Add random weights
        for edge in self.G.edges():
            self.G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10)
            
        # Rename nodes
        mapping = {i: chr(65 + i) for i in range(nodes)}  # A, B, C, ...
        self.G = nx.relabel_nodes(self.G, mapping)
        
        self.node_positions_fixed = False
        self.draw_graph()
        self.status_var.set(f"Created complete graph with {nodes} nodes")
        
    def clear_graph(self):
        self.G.clear()
        self.pos.clear()
        self.node_colors.clear()
        self.edge_colors.clear()
        self.highlighted_path = []
        self.node_positions_fixed = False
        self.draw_graph()
        self.status_var.set("Graph cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingVisualizer(root)
    root.mainloop()