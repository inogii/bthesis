import graph_tool.all as gt
import numpy as np
import json 
import pycountry_convert as country
import matplotlib.pyplot as plt

#global variables
GRAPH_PATH = "AS_graph.gt"
countries = set()
continents = set()
business = set()
relationships = set()
rirs = set()
not_int_props = {'node', 'link', 'link_nodes', 'node_ASN', 'node_org_name', 'node_rir', 'node_hq_country', 'node_hq_continent', 'node_business_type', 'node_is_VP', 'link_relationship', 'link_seeing_RCs'}

def load_graph():
    g = gt.load_graph(GRAPH_PATH)
    node_properties = g.vertex_properties
    link_properties = g.edge_properties
    nodes = g.vertices()
    links = g.edges()
    return nodes, links, node_properties, link_properties

def initialize_lists(nodes, node_properties, links, link_properties):

    global countries
    global continents
    global business
    global relationships
    global rirs

    for node in nodes:
        countries.add(node_properties['hq_country'][node])
        continents.add(node_properties['hq_continent'][node])
        business.add(node_properties['business_type'][node])
        rirs.add(node_properties['rir'][node])
    
    for link in links:
        relationships.add(link_properties['relationship'][link])

    countries = list(countries)
    continents = list(continents)
    business = list(business)
    relationships = list(relationships)
    rirs = list(rirs)


def build_dict(nodes, links, node_properties, link_properties):

    output_dict = {"node" : [], "link": []}
    count = 0

    for node in nodes:
        node_entry = {
            "node" : int(count)
        }
        for prop in node_properties:
            property_name = 'node_'+ str(prop)
            #Changing string properties to number
            if prop == 'business_type':
                node_entry.update({property_name : business.index(str(node_properties[prop][node]))})
            elif prop == 'hq_country':
                node_entry.update({property_name : countries.index(str(node_properties[prop][node]))})
            elif prop == 'hq_continent':
                node_entry.update({property_name : continents.index(str(node_properties[prop][node]))})
            elif prop == 'rir':
                 node_entry.update({property_name : rirs.index(str(node_properties[prop][node]))})
            #closeness_d skipped when not a number
            elif prop == 'closeness_d' and str(node_properties[prop][node]) == 'nan' :
                node_entry.update({property_name : 0})
            else:
                node_entry.update({property_name : node_properties[prop][node]})
        output_dict["node"].append(node_entry)
        count += 1 

    for link in links:
        link_nodes = tuple(link)
        link_nodes = [int(link_nodes[0]), int(link_nodes[1])]
        link_entry = {
            "link" : count,
            "link_nodes" : link_nodes
        }
        for prop in link_properties:
            property_name = 'link_'+ str(prop)

            if prop == 'seeing_RCs':
                continue
            #Changing string properties to number
            elif prop == 'relationship':
                link_entry.update({property_name : relationships.index(str(link_properties[prop][link]))})
            else:
                link_entry.update({property_name : link_properties[prop][link]})
        output_dict["link"].append(link_entry)
        count+=1
    
    return output_dict
    
# Normalizes the data in the output dictionary to be 0 to 1 using max_min normalization
def max_min_normalization(output_dict):

    max_property = {}
    min_property = {}
    
    for node in output_dict['node']:
        for prop in node.keys():
            if prop not in not_int_props:
                if prop in max_property.keys():
                    if node[prop] > max_property[prop]:
                        max_property[prop] = node[prop]
                else :
                    new_entry = {prop : node[prop]} 
                    max_property.update(new_entry)
                if prop in min_property.keys():
                    if node[prop] < min_property[prop]:
                        min_property[prop] = node[prop]
                else :
                    new_entry = {prop : node[prop]} 
                    min_property.update(new_entry)
    
    for node in output_dict['node']:
        for prop in node.keys():
            if prop not in not_int_props:
                value = node[prop]
                node[prop] = (value - min_property[prop])/(max_property[prop]-min_property[prop])

    for link in output_dict['link']:
        for prop in link.keys():
            if prop not in not_int_props:
                if prop in max_property.keys():
                    if link[prop] > max_property[prop]:
                        max_property[prop] = link[prop]
                else :
                    new_entry = {prop : link[prop]} 
                    max_property.update(new_entry)
                if prop in min_property.keys():
                    if link[prop] < min_property[prop]:
                        min_property[prop] = link[prop]
                else :
                    new_entry = {prop : link[prop]} 
                    min_property.update(new_entry)

    for link in output_dict['link']:
        for prop in link.keys():
            if prop not in not_int_props:
                value = link[prop]
                link[prop] = (value-min_property[prop])/(max_property[prop]-min_property[prop])
    
    return output_dict

#normalizes the data in the output dictionary by using the mean and std of the distribution of each property
#more robust against outliers
def z_normalization(output_dict):
    
    mean = {}
    std = {}
    node0 = output_dict['node'][0]
    link0 = output_dict['link'][0]
    
    for prop in node0.keys():
        if prop not in not_int_props:
            new_entry = {prop : 0} 
            mean.update(new_entry)
            std.update(new_entry)
            mean[prop] = np.mean([c[prop] for c in output_dict['node']])
            std[prop] = np.std([c[prop] for c in output_dict['node']])
    
    for prop in link0.keys():
        if prop not in not_int_props:
            new_entry = {prop : 0} 
            mean.update(new_entry)
            std.update(new_entry)
            mean[prop] = np.mean([c[prop] for c in output_dict['link']])
            std[prop] = np.std([c[prop] for c in output_dict['link']])

    for node in output_dict['node']:
        for prop in node.keys():
            if prop not in not_int_props:
                value = node[prop]
                node[prop] = (value - mean[prop])/std[prop]

    for link in output_dict['link']:
        for prop in link.keys():
            if prop not in not_int_props:
                value = link[prop]
                link[prop] = (value - mean[prop])/std[prop]

    return output_dict

def visualize_properties(output_dict):
    plt.rcParams["figure.figsize"] = [25, 25]
    node0 = output_dict['node'][0]
    link0 = output_dict['link'][0]
    fig, axs = plt.subplots(5, 5)
    count1=0
    count2=0
    for prop in node0.keys():
        if prop not in not_int_props:
            counts, bins = np.histogram(([c[prop] for c in output_dict['node']]), bins='auto', density=True)
            axs[count1, count2].hist(counts, bins)
            axs[count1, count2].set_title(prop)
            count2 +=1
            if count2>4:
                count1 +=1
                count2 = 0
            
    for prop in link0.keys():
        if prop not in not_int_props:
            counts, bins = np.histogram(([c[prop] for c in output_dict['link']]), bins='auto', density=True)
            axs[count1, count2].hist(counts, bins)
            axs[count1, count2].set_title(prop)
            count2 +=1
            if count2 >4:
                count1 +=1
                count2 = 0


def write_files(output_dict):

    output_nodes = {"node" : output_dict['node']}
    output_links = {"link" : output_dict['link']}
    text_file = open('graph_nodes.json', 'w')
    n = text_file.write(json.dumps(output_nodes))
    text_file.close()

    text_file = open('graph_links.json', 'w')
    n = text_file.write(json.dumps(output_links))
    text_file.close()

def write_properties():
    
    properties_dict = {
        "countries_len" : len(countries),
        "continents_len" : len(continents),
        "business_len" : len(business),
        "rirs_len" : len(rirs),
        "relationships_len" : len(relationships)
    }
    text_file = open('properties.json', 'w')
    n = text_file.write(json.dumps(properties_dict))
    text_file.close()

def main():
    
    nodes, links, node_properties, link_properties = load_graph()
    initialize_lists(nodes, node_properties, links, link_properties)
    write_properties()
    nodes, links, node_properties, link_properties = load_graph()
    output_dict = build_dict(nodes, links, node_properties, link_properties)
    output_dict = max_min_normalization(output_dict)
    visualize_properties(output_dict)
    plt.savefig('plot_minmax.pdf')
    write_files(output_dict)


# Execute program if not imported
if __name__ == "__main__":
    main()