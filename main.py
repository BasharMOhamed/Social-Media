import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import networkx as nx
import pandas as pd
from CentralityMeasures import calculate_centrality,draw_filtered_graph

class NetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Social Networks Analysis Tool")

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
        tk.Button(button_frame, text="Draw Graph", command="").pack(side=tk.LEFT, padx=5)

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
            values=("Fruchterman-Reingold"),
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
        filtered_G = self.G

        tk.Button(
            button_frame,
            text="Draw Filtered Graph",
            command=lambda: draw_filtered_graph(self.minCentrality, self.maxCentrality, filtered_G,
                                                self.centralityVar.get(), self.output_dir)
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            button_frame,
            text="Calculate Centrality",
            command=lambda: calculate_centrality(self.centralityVar.get(), filtered_G)
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
        tk.Button(link_analysis_frame,width=30, text="Link analysis", command=print("Link analysis")).grid(row=1,column=2, padx=5, pady=10)



















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

    def show_metrics(self):
        print(self.directness_var.get())
        if self.G is None:
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
        self.G = None

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