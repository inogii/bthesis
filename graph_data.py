import numpy as np
import graph_tool.all as gt
import json 
import pycountry_convert as country
import matplotlib.pyplot as plt
import pprint
import copy

#global variables
graph_path = "AS_graph.gt"
countries = set()
continents = set()
business = set()
relationships = set()
rirs = set()
not_int_props = {'node', 'link', 'link_nodes', 'node_ASN', 'node_org_name', 'node_rir', 'node_hq_country', 'node_hq_continent', 'node_business_type', 'node_is_VP', 'link_relationship', 'link_seeing_RCs'}
roles = ['NA', 'provider', 'customer', 'sibling', 'peer']

mode = 'default'

def init_global():
    global countries
    global continents
    global business
    global relationships
    global rirs
    countries = set()
    continents = set()
    business = set()
    relationships = set()
    rirs = set()

def load_graph():
    g = gt.load_graph(graph_path)
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

    output_dict = {"node" : [], "link": [], "role" : []}
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

        if mode == 'test':
            if count > 1:
                break

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
        if mode == 'test':
            if count > 2:
                break

    if mode != 'test':

        for elem in output_dict['link']:

            role1 = {
                    "role_number" : count,
                    "role_node" : elem['link_nodes'][0],
                    "role_link": elem['link'],
                    "role_role" : 'NA'
                    }
            
            count += 1

            role2 = {
                    "role_number" : count,
                    "role_node" : elem['link_nodes'][1],
                    "role_link": elem['link'],
                    "role_role" : 'NA'
                    }
            
            count += 1
            
            try:
                if elem['link_relationship'] == relationships.index('NA'):
                    role1['role_role'] = 'NA'
                    role2['role_role'] = 'NA'
                elif elem['link_relationship'] == relationships.index('s2s'):
                    role1['role_role'] = 'sibling'
                    role2['role_role'] = 'sibling'
                elif elem['link_relationship'] == relationships.index('p2c'):
                    role1['role_role'] = 'provider'
                    role2['role_role'] = 'customer'
                elif elem['link_relationship'] == relationships.index('c2p'):
                    role1['role_role'] = 'customer'
                    role2['role_role'] = 'provider'
                elif elem['link_relationship'] == relationships.index('p2p'):
                    role1['role_role'] = 'peer'
                    role2['role_role'] = 'peer'
            except ValueError:
                pass
            
            role1['role_role'] = roles.index(role1['role_role'])
            role2['role_role'] = roles.index(role2['role_role'])

            output_dict['role'].append(role1)
            output_dict['role'].append(role2)
            
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

def modify_countries(output_dict):
    global countries 
    countries_count = {}
    threshold = 250
    for node in output_dict['node']:
        country_number = node['node_hq_country']
        country_name = countries[country_number]
        if country_name in countries_count.keys():
            countries_count[country_name] += 1
        else:
            countries_count[country_name] = 1
    #sort the dictionary by value
    countries_count = sorted(countries_count.items(), key=lambda x:x[1])
    #print number of tuples in the dictionary that are lower than 100
    excluded_countries = dict([x for x in countries_count if x[1] < threshold])
    excluded_continent = {}
    for entry in excluded_countries:
        country_alpha2 = entry
        continent = country.country_alpha2_to_continent_code(country_alpha2)
        if 'rest_of_' + continent in excluded_continent.keys():
            excluded_continent['rest_of_'+continent] += excluded_countries[country_alpha2]
        else:
            excluded_continent['rest_of_'+continent] = excluded_countries[country_alpha2]
    
    included_countries = dict([x for x in countries_count if x[1] >= threshold])
    if 'NOT_AVAILABLE' in included_countries.keys():
        included_countries.pop('NOT_AVAILABLE')
    included_countries.update(excluded_continent)
    excluded_nodes = sum([x for x in excluded_countries.values()])
    excluded_cont_count = sum([x for x in excluded_continent.values()])
    
    text_file = open('country_distribution.json', 'w')
    n = text_file.write(json.dumps(included_countries))
    text_file.close()

    included_countries = included_countries.keys()
    for node in output_dict['node']:
        country_number = node['node_hq_country']
        country_alpha2 = countries[country_number]
        if country_alpha2 == 'NOT_AVAILABLE':
            node['node_hq_country'] = -1
        elif country_alpha2 in included_countries:
            node['node_hq_country'] = list(included_countries).index(country_alpha2)
        else:
            continent = country.country_alpha2_to_continent_code(country_alpha2)
            node['node_hq_country'] = list(included_countries).index('rest_of_'+continent)
    # for i in range(included_countries):
    #     if not included_countries[i].startswith('rest_of_'):
    #         elem = country.country_alpha2_to_country_name(elem)
    countries = list(included_countries)
    return output_dict

def modify_business(output_dict):
    global business
    ndiscl_index = business.index('Not Disclosed')
    new_business = ['Transit Access', 'Enterprise', 'Content']

    mapping = {"Network Services" : "Transit Access", 
               "Non-Profit": "Enterprise", 
               "Content": "Content", 
               "NSP": "Transit Access", 
               "Government" : "Enterprise", 
               "Enterprise" : "Enterprise", 
               "Route Collector" : "Transit Access", 
               "Educational/Research" : "Enterprise", 
               "Cable/DSL/ISP": "Transit Access"}
    
    for node in output_dict['node']:
        business_number = node['node_business_type']
        if business_number == ndiscl_index:
            node['node_business_type'] = -1
        else:
            old_label = business[business_number]
            new_label = mapping.get(old_label)
            new_index = new_business.index(new_label)
            node['node_business_type'] = new_index
    
    business = new_business
    return output_dict

def modify_rir(output_dict):
    global rirs
    ndiscl_index = rirs.index('NOT_AVAILABLE')
    new_rir = copy.copy(rirs)
    new_rir.remove('NOT_AVAILABLE')
    for node in output_dict['node']:
        rir_number = node['node_rir']
        if rir_number == ndiscl_index:
            node['node_rir'] = -1
        else:
            new_index = new_rir.index(rirs[rir_number])
            node['node_rir'] = new_index
    rirs = new_rir
    return output_dict

def modify_continent(output_dict):
    global continents
    ndiscl_index = continents.index('')
    new_continents = copy.copy(continents)
    new_continents.remove('')
    for node in output_dict['node']:
        continent_number = node['node_hq_continent']
        if continent_number == ndiscl_index:
            node['node_hq_continent'] = -1
        else:
            new_index = new_continents.index(continents[continent_number])
            node['node_hq_continent'] = new_index
    continents = new_continents
    return output_dict

def modify_relationship(output_dict):
    global relationships
    ndiscl_index = relationships.index('NA')
    new_relationship = copy.copy(relationships)
    new_relationship.remove('NA')
    for link in output_dict['link']:
        relationship_number = link['link_relationship']
        if relationship_number == ndiscl_index:
            link['link_relationship'] = -1
        else:
            new_index = new_relationship.index(relationships[relationship_number])
            link['link_relationship'] = new_index
    relationships = new_relationship
    return output_dict


def write_files(output_dict):
    
    output_nodes = {"node" : output_dict['node']}
    output_links = {"link" : output_dict['link']}
    output_roles = {"role" : output_dict['role']}

    text_file = open('graph_nodes.json', 'w')
    n = text_file.write(json.dumps(output_nodes))
    text_file.close()

    text_file = open('graph_links.json', 'w')
    n = text_file.write(json.dumps(output_links))
    text_file.close()

    text_file = open('graph_roles.json', 'w')
    n = text_file.write(json.dumps(output_roles))
    text_file.close()

def write_properties():
    
    properties_dict = {
        "countries_len" : len(countries),
        "continents_len" : len(continents),
        "business_len" : len(business),
        "rirs_len" : len(rirs),
        "relationships_len" : len(relationships),
        "roles_len" : len(roles),
        "business_classification" : business,
        "continent_classification" : continents,
        "country_classification" : countries,
        "rir_classification" : rirs,
        "link_classification" : relationships
    }
    text_file = open('properties.json', 'w')
    n = text_file.write(json.dumps(properties_dict))
    text_file.close()

def create_jsons(normalization='minmax', running_mode='default'):
    
    global mode
    global graph_path

    mode = running_mode

    if mode == 'test':
        graph_path = 'AS_graph_test.gt'
    else:
        graph_path = 'AS_graph.gt'

    normalization_options = ['none', 'minmax', 'z']
    if normalization not in normalization_options:
        print("Invalid option. Please choose between 'none', 'minmax' and 'z'")
        return
    
    init_global()
    nodes, links, node_properties, link_properties = load_graph()
    initialize_lists(nodes, node_properties, links, link_properties)
    nodes, links, node_properties, link_properties = load_graph()
    output_dict = build_dict(nodes, links, node_properties, link_properties)
    if mode != 'test':
        if normalization == 'none':
            print("No normalization selected")
        if normalization == 'z':
            print("Z normalization selected")
            output_dict = z_normalization(output_dict)
        if normalization == 'minmax':
            print("Minmax normalization selected")
            output_dict = max_min_normalization(output_dict)
    output_dict = modify_countries(output_dict) 
    output_dict = modify_business(output_dict)
    output_dict = modify_rir(output_dict)
    output_dict = modify_continent(output_dict)
    output_dict = modify_relationship(output_dict)
    write_files(output_dict)
    write_properties()
    return 1

# Execute program if not imported
if __name__ == "__main__":
    create_jsons(normalization='minmax', running_mode='test')
