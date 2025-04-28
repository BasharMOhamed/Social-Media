import networkx as nx
import csv

degree_centrality = ''
closeness_centrality = ''
betweenness_centrality = ''


def calculate_centrality(centrality_type, G, path):
    print('here')
    print(centrality_type)
    print(G)
    global degree_centrality, closeness_centrality, betweenness_centrality
    if centrality_type == 'Degree':
        degree_centrality = nx.degree_centrality(G)
        print (degree_centrality)
        write_centrality_in_file(degree_centrality,path)
        return degree_centrality
    elif centrality_type == 'Closeness':
        closeness_centrality = nx.closeness_centrality(G)
        print(closeness_centrality)
        write_centrality_in_file(closeness_centrality, path)
        return closeness_centrality
    elif centrality_type == 'Betwenness':
        betweenness_centrality = nx.betweenness_centrality(G)
        print(betweenness_centrality)
        write_centrality_in_file(betweenness_centrality, path)
        return betweenness_centrality


def draw_filtered_graph(min_val, max_val, G, centrality_type):
    if centrality_type == 'Degree' and degree_centrality != '':
        for node, centrality in degree_centrality.items():
            if centrality < min_val & centrality > max_val:
                G.remove_node(node)

    if centrality_type == 'Closeness' and closeness_centrality != '':
        for node, centrality in closeness_centrality.items():
            if centrality < min_val & centrality > max_val:
                G.remove_node(node)

    if centrality_type == 'Betwenness' and betweenness_centrality != '':
        for node, centrality in betweenness_centrality.items():
            if centrality < min_val & centrality > max_val:
                G.remove_node(node)


def write_centrality_in_file(centrality,path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(centrality)

    print(f"CSV file created at {path}")
