import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from CentralityMeasures import calculate_centrality, draw_filtered_graph, filtered_G, centralities
class NetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Social Networks Analysis Tool")
        self.output_dir = os.path.join(os.getcwd(),'centrality.csv')

        # Graph
        self.G = None
        self.nodes_df = None
        self.edges_df = None

        # GUI Layout
        self.create_widgets()

    def create_widgets(self):
        # Main container frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Top row buttons - File operations
        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding=10)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        button_frame = tk.Frame(file_frame)
        button_frame.pack(fill=tk.X)

        tk.Button(button_frame, text="Load Nodes CSV", command=self.load_nodes).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Load Edges CSV", command=self.load_edges).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Draw Graph", command=self.draw_graph).pack(side=tk.LEFT, padx=5)

        # Status bar at bottom
        self.status_label = tk.Label(file_frame, text="Ready", fg="blue", anchor="w")
        self.status_label.pack(fill=tk.X, pady=(5,0))

        # Graph properties frame
        prop_frame = ttk.LabelFrame(main_frame, text="Graph Properties", padding=10)
        prop_frame.grid(row=1, column=0, sticky="w", pady=5)

        # Directness selection
        ttk.Label(prop_frame, text="Graph Type:").grid(row=0, column=0, sticky="e")
        self.directness_var = tk.StringVar(value="Undirected")
        self.directness_menu = ttk.Combobox(
            prop_frame,
            textvariable=self.directness_var,
            values=("Undirected", "Directed"),
            state="readonly",
            width=15
        )
        self.directness_menu.grid(row=0, column=1, padx=5, sticky="w")

        # Layout algorithm selection
        ttk.Label(prop_frame, text="Layout Algorithm:").grid(row=0, column=2, sticky="e")
        self.layout_var = tk.StringVar(value="Fruchterman-Reingold")
        self.layout_menu = ttk.Combobox(
            prop_frame,
            textvariable=self.layout_var,
            values=("Simple","Fruchterman-Reingold","Tree (Hierarchical)","Radial Layout"),
            state="readonly",
            width=20
        )
        self.layout_menu.grid(row=0, column=3, padx=5, sticky="w")


        # Configure grid weights for proper resizing
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)



        centrality_frame = tk.LabelFrame(self.root, text="Centrality", padx=5, pady=5)
        centrality_frame.pack(padx=10, pady=5, fill=tk.X)


#########################################################################################################################


        # Centrality Frame
        centrality_frame = ttk.LabelFrame(self.root, text="Centrality Filter", padding=10)
        centrality_frame.pack(fill=tk.X, padx=10, pady=5)

        # First Row: Dropdown Menu
        dropdown_frame = ttk.Frame(centrality_frame)
        dropdown_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(dropdown_frame, text="Centrality:").pack(side=tk.LEFT)
        self.centralityVar = tk.StringVar(value="Degree")
        self.centrality_dropdown = ttk.Combobox(
            dropdown_frame,
            textvariable=self.centralityVar,
            width=15,
            state="readonly"
        )
        self.centrality_dropdown['values'] = ('Degree', 'Closeness', 'Betweenness')
        self.centrality_dropdown.pack(side=tk.LEFT, padx=5)

        # Second Row: Min/Max Inputs
        range_frame = ttk.Frame(centrality_frame)
        range_frame.pack(fill=tk.X, pady=(0, 10))

        # Min Value
        ttk.Label(range_frame, text="Min:").pack(side=tk.LEFT)
        self.minCentrality = tk.DoubleVar(value=0.0)
        min_entry = ttk.Entry(
            range_frame,
            textvariable=self.minCentrality,
            width=8,
            validate="key"
        )
        min_entry.pack(side=tk.LEFT, padx=5)

        # Max Value
        ttk.Label(range_frame, text="Max:").pack(side=tk.LEFT)
        self.maxCentrality = tk.DoubleVar(value=20.0)
        max_entry = ttk.Entry(
            range_frame,
            textvariable=self.maxCentrality,
            width=8,
            validate="key"
        )
        max_entry.pack(side=tk.LEFT, padx=5)


        # Third Row: Buttons
        button_frame = ttk.Frame(centrality_frame)
        button_frame.pack(fill=tk.X)

        tk.Button(
            button_frame,
            text="Draw Filtered Graph",
            command=self.draw_centralities,
            # command=lambda: draw_filtered_graph(self.minCentrality.get(), self.maxCentrality.get(), self.G, self.centralityVar.get())
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            button_frame,
            text="Calculate Centrality",
            command= self.calculate_centrality_safe()
            # command=lambda: calculate_centrality(self.G, self.output_dir)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            button_frame,
            text="Select Output Directory",
            command=self.select_output_dir
        ).pack(side=tk.LEFT, padx=2)


  ##################################################################################################################
        clustering_frame = tk.LabelFrame(self.root, text="Clustering", padx=5, pady=5)
        clustering_frame.pack(padx=10, pady=5, fill=tk.X)

        # Centrality DropMenu
        tk.Label(clustering_frame, text="Clutering Algo:").grid(row=0, column=0, padx= 5)
        self.clusteringVar = tk.StringVar(value="Grivan-Newman")
        self.clustering_dropdown = ttk.Combobox(clustering_frame, textvariable=self.clusteringVar, width=25)
        self.clustering_dropdown['values'] = (
            'Grivan-Newman',
        )
        self.clustering_dropdown.grid(column=1, row=0 , padx=5)

        tk.Button(clustering_frame, text="Draw Clustered Graph", command="").grid(row=0,column=2, padx=5, pady=10)


        tk.Button(clustering_frame, text="Calculate NMI", command="").grid(row=1,column=0, padx=5, pady=10)
        tk.Button(clustering_frame, text="Calculate Conductance", command="").grid(row=1,column=1, padx=5, pady=10)
        tk.Button(clustering_frame, text="Calculate Modularity", command="").grid(row=1,column=2, padx=5, pady=10)

####################################################################################################################################

        link_analysis_frame = tk.LabelFrame(self.root, text="Link Analysis", padx=5, pady=5)
        link_analysis_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Button(link_analysis_frame,width=30, text="Calculate PageRank", command=self.calculate_pagerank).grid(row=1,column=2, padx=5, pady=10)


        #############################################################################################################
        # Action buttons
        action_frame = tk.Frame(main_frame)
        action_frame.grid(row=3, column=0, sticky="w", pady=10)

        tk.Button(action_frame, text="Show Metrics", command=self.show_metrics).pack(side=tk.LEFT, padx=5)

        # Customization frames - using Notebook for better organization
        custom_notebook = ttk.Notebook(main_frame)
        custom_notebook.grid(row=4, column=0, sticky="nsew", pady=10)

        # Node customization tab
        node_frame = ttk.Frame(custom_notebook)
        custom_notebook.add(node_frame, text="Node Style")

        tk.Label(node_frame, text="Size:").grid(row=0, column=0, sticky="e")
        self.node_size = tk.IntVar(value=50)
        tk.Scale(node_frame, from_=10, to=500, variable=self.node_size, orient=tk.HORIZONTAL).grid(row=0, column=1)

        # Add other node customization controls similarly...
        # Node Size
        tk.Label(node_frame, text="Size:").grid(row=0, column=0)
        self.node_size = tk.IntVar(value=50)
        tk.Scale(node_frame, from_=10, to=500, variable=self.node_size, orient=tk.HORIZONTAL).grid(row=0, column=1)

        # Node Color
        tk.Label(node_frame, text="Color:").grid(row=1, column=0)
        self.node_color = tk.StringVar(value="skyblue")
        ttk.Combobox(node_frame, textvariable=self.node_color,
                     values=("skyblue", "red", "green", "yellow", "orange", "purple")).grid(row=1, column=1)

        # Node Shape
        tk.Label(node_frame, text="Shape:").grid(row=2, column=0)
        self.node_shape = tk.StringVar(value="o")
        ttk.Combobox(node_frame, textvariable=self.node_shape,
                     values=("o", "s", "^", "v", "<", ">", "8", "p", "h", "H", "D", "d")).grid(row=2, column=1)

        # Edge customization tab
        edge_frame = ttk.Frame(custom_notebook)
        custom_notebook.add(edge_frame, text="Edge Style")

        # Add edge controls...
        # Edge Width
        tk.Label(edge_frame, text="Width:").grid(row=0, column=0)
        self.edge_width = tk.IntVar(value=1)
        tk.Scale(edge_frame, from_=1, to=10, variable=self.edge_width, orient=tk.HORIZONTAL).grid(row=0, column=1)

        # Edge Color
        tk.Label(edge_frame, text="Color:").grid(row=1, column=0)
        self.edge_color = tk.StringVar(value="gray")
        ttk.Combobox(edge_frame, textvariable=self.edge_color,
                     values=("gray", "black", "red", "blue", "green")).grid(row=1, column=1)

        # Edge Style
        tk.Label(edge_frame, text="Style:").grid(row=2, column=0)
        self.edge_style = tk.StringVar(value="solid")
        ttk.Combobox(edge_frame, textvariable=self.edge_style,
                     values=("solid", "dashed", "dotted", "dashdot")).grid(row=2, column=1)

        # Label customization tab
        label_frame = ttk.Frame(custom_notebook)
        custom_notebook.add(label_frame, text="Labels")

        # Add label controls...
        tk.Label(label_frame, text="Show Labels:").grid(row=0, column=0)
        self.show_labels = tk.BooleanVar(value=True)
        tk.Checkbutton(label_frame, variable=self.show_labels).grid(row=0, column=1)

        tk.Label(label_frame, text="Label Color:").grid(row=1, column=0)
        self.label_color = tk.StringVar(value="black")
        ttk.Combobox(label_frame, textvariable=self.label_color,
                     values=("black", "red", "blue", "green", "white")).grid(row=1, column=1)

        tk.Label(label_frame, text="Label Size:").grid(row=2, column=0)
        self.label_size = tk.IntVar(value=8)
        tk.Scale(label_frame, from_=6, to=20, variable=self.label_size, orient=tk.HORIZONTAL).grid(row=2, column=1)



        # Configure grid weights
        main_frame.grid_rowconfigure(4, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def load_nodes(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.nodes_df = pd.read_csv(file_path)
            self.status_label.config(text=f"Loaded {len(self.nodes_df)} nodes.", fg="green")

    def load_edges(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.edges_df = pd.read_csv(file_path)
            self.status_label.config(text=f"Loaded {len(self.edges_df)} edges.", fg="green")




    def draw_graph(self):
        if not hasattr(self, 'nodes_df') or not hasattr(self, 'edges_df') or self.nodes_df is None or self.edges_df is None:
            self.status_label.config(text="No graph data loaded!", fg="red")
            return

        
        graph_type = self.directness_var.get()
        G = nx.DiGraph() if graph_type == "Directed" else nx.Graph()

        labels = {}
        # Add nodes & edges
        for _, row in self.nodes_df.iterrows():
            G.add_node(row['ID'])
            labels[row['ID']] = str(row['ID'])

        for _, row in self.edges_df.iterrows():
            G.add_edge(row['Source'], row['Target'])
        
        self.G = G

        # Layout selection
        layout_choice = self.layout_var.get()
        if layout_choice == "Fruchterman-Reingold":
            pos = nx.spring_layout(G)
        elif layout_choice == "Tree (Hierarchical)" :
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                print("Graphviz not installed, using spring layout as fallback.")
                pos = nx.spring_layout(G)
        elif layout_choice == "Radial Layout" :
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Graph options
        node_size = self.node_size.get()
        node_color = self.node_color.get()
        node_shape = self.node_shape.get()

        edge_width = self.edge_width.get()
        edge_color = self.edge_color.get()
        edge_style = self.edge_style.get()

        show_labels = self.show_labels.get()  
        label_color = self.label_color.get()
        label_size = self.label_size.get()


        plt.figure(figsize=(10, 8))
        

        nx.draw_networkx_nodes(G, pos,
                            node_size=node_size,
                            node_color=node_color,
                            node_shape=node_shape)


        nx.draw_networkx_edges(G, pos,
                            width=edge_width,
                            edge_color=edge_color,
                            style=edge_style,
                            alpha=0.2)


        if show_labels:
            nx.draw_networkx_labels(G, pos,
                                    labels=labels,
                                    font_size=label_size,
                                    font_color=label_color)

        plt.axis('off')
        plt.title(f"{layout_choice} Graph Visualization")
        plt.show()

        self.status_label.config(text="Graph drawn successfully!", fg="green")


    def calculate_pagerank(self):
        if not hasattr(self, 'nodes_df') or not hasattr(self, 'edges_df') or self.nodes_df is None or self.edges_df is None:
            messagebox.showerror("Error", "No graph data loaded!")
            return

        # Create the graph
        graph_type = self.directness_var.get()
        if graph_type == "Directed":
            G = nx.DiGraph()
        else:
            G = nx.Graph()


        for _, row in self.nodes_df.iterrows():
            G.add_node(row['ID'])

        for _, row in self.edges_df.iterrows():
            G.add_edge(row['Source'], row['Target'])

        # Calculate PageRank
        try:
            pagerank_results = nx.pagerank(G)
        except nx.PowerIterationFailedConvergence:
            messagebox.showerror("Error", "PageRank did not converge!")
            return

        # Create a new window to display results
        result_window = tk.Toplevel()
        result_window.title("PageRank Results")

        text_widget = tk.Text(result_window, wrap="word", width=60, height=20)
        text_widget.pack(fill="both", expand=True)

        pagerank_text = "\n".join(f"Node {node}: {rank:.4f}" for node, rank in pagerank_results.items())
        text_widget.insert("1.0", pagerank_text)

        text_widget.config(state="disabled")

    
    def calculate_centrality_safe(self):
        if not hasattr(self, 'G') or self.G is None:
            self.status_label.config(text="No graph loaded. Please draw a graph first.", fg="red")
            return
        calculate_centrality(self.G, self.output_dir)
        self.status_label.config(text="Centrality calculated successfully!", fg="green")


    def draw_centralities(self):
        if not hasattr(self, 'G') or self.G is None:
            self.status_label.config(text="No graph loaded. Please draw a graph first.", fg="red")
            return
        
        centralities = calculate_centrality(self.G, self.output_dir)
        filtered_G = draw_filtered_graph(self.minCentrality.get(), self.maxCentrality.get(), self.G, self.centralityVar.get())
        if filtered_G is None or filtered_G.number_of_nodes() == 0:
            self.status_label.config(text="No graph to display. Make sure to calculate centralities first.", fg="red")
            return


        graph_type = self.directness_var.get()
        G = nx.DiGraph() if graph_type == "Directed" else nx.Graph()
        # Add only existing nodes and edges from filtered_G
        G.add_nodes_from(filtered_G.nodes(data=True))
        G.add_edges_from(filtered_G.edges(data=True))

        labels = {node: str(node) for node in G.nodes()}


        # Graph options
        node_color = self.node_color.get()
        node_shape = self.node_shape.get()

        edge_width = self.edge_width.get()
        edge_color = self.edge_color.get()
        edge_style = self.edge_style.get()

        show_labels = self.show_labels.get()
        label_color = self.label_color.get()
        label_size = self.label_size.get()

        #node sizes based on centralities values
        centrality_type = self.centralityVar.get()
        centrality_index = {"Degree Centrality": 1, "Closeness Centrality": 2, "Betweenness Centrality": 3}
        index = centrality_index.get(centrality_type, 1)
        centrality_dict = {entry[0]: entry[index] for entry in centralities if entry[0] in G.nodes()}
        
        min_size = 50  
        max_size = 800 

        centrality_values = list(centrality_dict.values())

        min_c = min(centrality_values)
        max_c = max(centrality_values)

        node_sizes = [
            min_size + (centrality_dict[node] - min_c) / (max_c - min_c) * (max_size - min_size)
            for node in G.nodes()
        ]

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        nx.draw_networkx_nodes(G, pos,
                                node_size=node_sizes,
                                node_color=node_color,
                                node_shape=node_shape)

        nx.draw_networkx_edges(G, pos,
                                width=edge_width,
                                edge_color=edge_color,
                                style=edge_style,
                                alpha=0.2)

        if show_labels:
            nx.draw_networkx_labels(G, pos,
                                    labels=labels,
                                    font_size=label_size,
                                    font_color=label_color)

        plt.axis('off')
        plt.title(f"{self.centralityVar.get()} Filtered Graph Visualization")
        plt.show()

        self.status_label.config(text="Filtered Graph drawn successfully!", fg="green")

    def show_metrics(self):
        print(self.directness_var.get())
        if self.nodes_df is not None and self.edges_df is not None:
                if self.directness_var.get() == "Directed":
                    self.G = nx.DiGraph()
                else:
                    self.G = nx.Graph()

                    # Add nodes and edges
                for _, row in self.nodes_df.iterrows():
                    self.G.add_node(row['ID'])
                for _, row in self.edges_df.iterrows():
                    self.G.add_edge(row['Source'], row['Target'])
        else:
            messagebox.showerror("Error", "Graph not created yet.")
            return

        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        metrics = [
            f"Graph Type: {'Directed' if nx.is_directed(self.G) else 'Undirected'}",
            f"Nodes: {num_nodes}",
            f"Edges: {num_edges}"
        ]

        # Calculate degree metrics
        degrees = [d for n, d in self.G.degree()]
        avg_degree = sum(degrees) / num_nodes
        metrics.append(f"Average Degree: {avg_degree:.2f}")

        # For directed graphs, calculate in/out degree metrics
        if nx.is_directed(self.G):
            in_degrees = [d for n, d in self.G.in_degree()]
            out_degrees = [d for n, d in self.G.out_degree()]
            avg_in_degree = sum(in_degrees) / num_nodes
            avg_out_degree = sum(out_degrees) / num_nodes
            metrics.extend([
                f"In-Degree: {sum(in_degrees)}",
                f"Out-Degree: {sum(out_degrees)}",
                f"Average In-Degree: {avg_in_degree:.2f}",
                f"Average Out-Degree: {avg_out_degree:.2f}"
            ])

        # Calculate clustering coefficient (only for undirected graphs)
        if not nx.is_directed(self.G):
            avg_clustering = nx.average_clustering(self.G)
            metrics.append(f"Average Clustering: {avg_clustering:.2f}")

        # Calculate average path length (if graph is connected)
        try:
            avg_path_length = nx.average_shortest_path_length(self.G)
            metrics.append(f"Average Path Length: {avg_path_length:.2f}")
        except nx.NetworkXError:
            metrics.append("Average Path Length: Graph not connected")

        # Calculate density
        density = nx.density(self.G)
        metrics.append(f"Density: {density:.4f}")

        # Show metrics in messagebox
        messagebox.showinfo("Graph Metrics", "\n".join(metrics))

    def select_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir = dir_path
            self.status_label.config(text=f"Output directory: {dir_path}", fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkApp(root)
    root.mainloop()