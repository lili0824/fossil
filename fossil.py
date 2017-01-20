from ete3 import Tree, TreeStyle
import scipy.spatial.distance
import numpy as np
from sympy import *
from scipy import *
import os

# the actual tree
# internal nodes
a,b,c,d,e,f,g,h,i,j,k,l = symbols('a,b,c,d,e,f,g,h,i,j,k,l')
# leaf nodes
mandrillus,cercocebus,lopocebus,papio,theropithecus,macaca,allenopithecus,chlorocebus,cercopithecus,miopithecus,presbytini,pygathrix,colobini = symbols('mandrillus,cercocebus,lopocebus,papio,theropithecus,macaca,allenopithecus,chlorocebus,cercopithecus,miopithecus,presbytini,pygathrix,colobini')
# tree structure
t_real = Tree("(((((mandrillus:3.1,cercocebus:3.1)a:4,(lopocebus:4.1,(papio:4,theropithecus:4)c:0.1)b:3)d:2,macaca:9.1)e:2.5,(allenopithecus:9.3,(chlorocebus:8.1,(cercopithecus:8,miopithecus:8)h:0.1)g:1.2)f:2.3)k:4.5,((presbytini:8.8,pygathrix:8.8)i:1.2,colobini:11)j:5)l;",format=1)

# input weights of all given leaf nodes to the weights dictionary
mydir = '/Users/lili/Documents/fossil_insert/landmark'
weights = {}
listing = os.listdir(mydir)
for infile in listing:
	if (infile.startswith('.') == False):
		#print("current file is: " , infile)
		data = np.loadtxt(mydir+'/'+infile,delimiter=' ')
		# conver to 1-d array
		data1 = data.flatten()
		filename = os.path.splitext(os.path.basename(infile))[0]
		weights[filename]=data1
		vec_len = len(weights[filename])

# store vectors to internal nodes
ref_table = {}
flag = 0
for node in t_real.traverse("levelorder"):
    if node.is_leaf() == False:
        ref_table[node.name] = flag
        flag = flag + 1

# update weights of leaf nodes
for leaf in t_real.traverse("levelorder"):
    leaf.add_features(weight=weights.get(leaf.name, "none"))
#print(t_real.get_ascii(attributes=["name", "weight"], show_internal=True))
#print(t_real.get_ascii(attributes=["name"], show_internal=True))

# solve for internal nodes
# create empty matrix to store values for solver
A1 = [[0 for i in range(flag)] for i in range(flag)]
B1 = [0 for i in range(flag)]

i = 0
for node in t_real.traverse("levelorder"):
    if node.is_leaf() == False:
        #print(node.name)
        sum_dist = 0
        p_dist = 0
        parent = 0
        if node.up != None:
            parent = node.up.weight
            p_dist = 1/node.dist
            A1[i][ref_table.get(node.up.name)] = -(p_dist)
        children_dist = 0
        children_weight = 0
        children_weight1 = 0
        for child in node.get_children():
            if child.is_leaf() == False:
                A1[i][ref_table.get(child.name)] = -(1/t_real.get_distance(node,child))
            else:
                child_dist1 = 1/t_real.get_distance(node,child)
                children_weight1 = children_weight1 + child_dist*child.weight
            B1[i] = children_weight1
            child_dist = 1/t_real.get_distance(node,child)
            children_dist = children_dist + child_dist
        sum_dist = children_dist + p_dist
        A1[i][ref_table.get(node.name)] = sum_dist
        i = i+1
# update the integer 0 to vector with 0s
for x in range(len(B1)):
    if (type(B1[x]) == int):
        B1[x] = np.array([0 for i in range(vec_len)])

A = np.array(A1)
B = np.array(B1)
result = np.linalg.solve(A,B)
#print(result)

# update tree internal node weight
t_result = t_real.copy("deepcopy")
for node in t_result.traverse("levelorder"):
    if node.is_leaf() == False:
        node.add_features(weight=result[ref_table.get(node.name)])

# verify the results
for node in t_result.traverse("levelorder"):
    if node.is_leaf() == False:
        sum_dist = 0.0
        p_dist = 0.0
        parent = 0.0
        # if the node is not root
        if node.up != None:
            parent = node.up.weight
            p_dist = 1/node.dist
        children_dist = 0.0
        children_weight = 0.0
        for child in node.get_children():
            child_dist = 1/t_result.get_distance(node,child)
            children_dist = children_dist + child_dist
            children_weight = children_weight + child_dist*child.weight
        sum_dist = children_dist + p_dist
        left = sum_dist*node.weight
        right = children_weight + p_dist*parent
        check = left - right
        #print("check",np.mean(check))

# update the branch lengths from time to morphological changes
for node in t_result.iter_descendants("postorder"):
    node_dist = scipy.spatial.distance.euclidean(node.weight,node.up.weight)
    node.dist = float(round(node_dist,3))
    #print("node - parent dist: ", node.dist)

# calculate the euclidean distance from a random data point f to an edge
f = np.array([1 for i in range(vec_len)])

# dictionary that saves the shortest euclidean distance from
# foreign fossil to an edge
distance2 = {}
#t2 is any real number
t2=1.5
for node in t_result.traverse("levelorder"):
    if node.is_leaf() == False:
        # calculate d as the shortest euclidean distance from random point f to an edge
        for child in node.get_children():
            d = 0
            #edge = node.name+"-"+child.name
            # add the Euclidean distance to the child end of an edge
            edge = child.name
            p_nodefossil = f - node.weight
            p_nodechild = child.weight - node.weight
            t = np.dot(p_nodefossil,p_nodechild)/np.dot(p_nodechild,p_nodechild)
            # calculate the projection of foreign fossil onto the edge
            #p_base = node.weight + t*(p_nodechild)
            p_base = t*p_nodechild
            # calculate the Euclidean distance from foreign fossil to edge
            d = scipy.spatial.distance.euclidean(f,p_base)
            distance2[edge] = float(round(d,5))

            # generalize the distance h from foreign fossil to an edge
            h = scipy.spatial.distance.euclidean(f,t2*p_nodechild)
            
print(distance2)
shortest = min(distance2, key=distance2.get)
print("Fossil is closest to branch:",shortest,"with Euclidean distance:",distance2[shortest])

# calculate the euclidean distance from foreign fossil to all nodes in tree
e_distance = {}
for node in t_result.traverse("levelorder"):
    euclidean_dist = scipy.spatial.distance.euclidean(f,node.weight)
    e_distance[node.name] = float(round(euclidean_dist,3))

# add e_distance as attribute to the tree,
# representing the euclidean distance between any given node
# in the tree and the foreign fossil
for node in t_result.traverse("levelorder"):
    node.add_features(e_dist=e_distance.get(node.name, "none"))
print(t_result.get_ascii(attributes=["name", "e_dist"], show_internal=True))

# add distance2 as attribute to the tree,
# representing the shortest euclidean distance
# from the foreign fossil to any given edge,
# an edge is denoted by the child-end node name
for node in t_result.traverse("levelorder"):
    node.add_features(e_dist2=distance2.get(node.name, "none"))
print(t_result.get_ascii(attributes=["name", "e_dist2"], show_internal=True))

t_out = t_result.write(format=1, features=["e_dist2"])
t_out1 = t_result.write(format=1)
print(t_out1)
t_result.write(format=1, features=["e_dist2"], outfile="real_tree.nw")


# level 1, find the projection s of foreign fossil f on an edge
t_level1 = t_result.copy("deepcopy")
child_node1 = t_level1.search_nodes(name=shortest)[0]
parent_node1 = child_node1.up
v = child_node1.weight - parent_node1.weight
w = f - parent_node1.weight
t = dot(w,v)/dot(v,v)
# level 1 s
s1 = t*v
s1 = parent_node1.add_child(name="s1")
foreign1 = s1.add_child(name="foreign")
removed1 = child_node1.detach()
s1.add_child(removed1)
print(t_level1.get_ascii(attributes=["name"], show_internal=True))
#print(parent_node.name)


# level 2, find s where s = 1/3(y+z+f)
t_level2 = t_result.copy("deepcopy")
child_node2 = t_level2.search_nodes(name=shortest)[0]
parent_node2 = child_node2.up
s2 = (child_node2.weight + parent_node2.weight + f)/3
s2 = parent_node2.add_child(name="s2")
foreign2 = s2.add_child(name="foreign")
removed2 = child_node2.detach()
s2.add_child(removed2)
print(t_level2.get_ascii(attributes=["name"], show_internal=True))








