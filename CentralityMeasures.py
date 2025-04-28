import networkx as nx
import csv

centralities = []


def calculate_centrality(G, path):
    global centralities

    degree_centrality = nx.degree_centrality(G)

    closeness_centrality = nx.closeness_centrality(G)

    betweenness_centrality = nx.betweenness_centrality(G)
    for node in G:
        centralities.append([
            node,
            degree_centrality[node],
            closeness_centrality[node],
            betweenness_centrality[node]
        ])

    write_centrality_in_file(path)


def draw_filtered_graph(min_val, max_val, G, centrality_type):
    filtered_G = G

    if centrality_type == 'Degree' and centralities != []:
        for subArray in centralities:
            if min_val > subArray[1] > max_val:
                filtered_G.remove_node(subArray[0])

    if centrality_type == 'Closeness' and centralities != []:
        for subArray in centralities:
            if min_val > subArray[2] > max_val:
                filtered_G.remove_node(subArray[0])

    if centrality_type == 'Betwenness' and centralities != []:
        for subArray in centralities:
            if min_val > subArray[3] > max_val:
                filtered_G.remove_node(subArray[0])


def write_centrality_in_file(path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Nodes", 'Degree_centrality', "Closeness_centrality", 'Betweenness_centrality'])
        writer.writerows(centralities)

    print(f"CSV file created at {path}")
