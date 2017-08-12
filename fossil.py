from ete3 import Tree, TreeStyle
import scipy.spatial.distance
import numpy as np
import os
import sys
import shutil
from collections import OrderedDict
import argparse

# define layout to draw trees
def my_layout(node):
    F = TextFace(node.name, tight_text=True)
    if node.is_leaf():
        add_face_to_node(F, node, column=0, position="branch-right")
    else:
        add_face_to_node(F, node, column=0, position="branch-bottom")

# a wrapper to calculate the distance between two vectors in space
# this function is made so later the Euclidean distance can be replaced
# by Mahalanobis distance
def distance(f1, f2):
	d = scipy.spatial.distance.euclidean(f1, f2)
	return d

# load original tree file
def read_tree(in_file):
    with open(in_file,'r') as file_ptr:
        in_tree = file_ptr.read()
    out_tree = Tree(in_tree,format=1)
    return out_tree

def get_name(base):
	name = os.path.splitext(base)
	return name[0]

# load the leaf landmark weights to the initial tree
def add_landmarkWeight(t,landmarks):
    weights = {}
    listing = os.listdir(landmarks)
    for infile in listing:
        if(infile.startswith('.') == False):
            data = np.loadtxt(landmarks+'/'+infile,delimiter=' ')
            filename = os.path.splitext(os.path.basename(infile))[0]
            weights[filename]=data.flatten()
    for leaf in t.traverse("levelorder"):
        leaf.add_features(weight=weights.get(leaf.name,"none"))
    return t

# return the length of landmark vectors
def landmark_len(t):
    w = {}
    for node in t.traverse("levelorder"):
        if (node.is_leaf() == True):
            w[node.name] = node.weight
    len_w = [len(x) for x in w.values()]
    if (len(set(len_w)) == 1):
        return len_w[0]

# solve for internal nodes and update tree with new weights
def weight_solver(t):
    ref_table = {}
    flag = 0
    for node in t.traverse("levelorder"):
        if node.is_leaf() == False:
            ref_table[node.name] = flag
            flag = flag + 1
    A1 = [[0 for i in range(flag)] for i in range(flag)]
    B1 = [0 for i in range(flag)]
    i = 0
    for node in t.traverse("levelorder"):
        if node.is_leaf() == False:
            sum_dist = 0
            p_dist = 0
            parent = 0
            children_dist = 0
            children_weight = 0
            children_weight1 = 0
            if node.up != None:
                parent = node.up.weight
                p_dist = 1/node.dist
                A1[i][ref_table[node.up.name]] = -(p_dist)
            for child in node.get_children():
                if child.is_leaf() == False:
                    A1[i][ref_table[child.name]] = -(1/t.get_distance(node,child))
                elif child.is_leaf() == True:
                    child_dist1 = 1/t.get_distance(node,child)
                    children_weight1 = children_weight1 + child_dist1*child.weight
                child_dist = 1/t.get_distance(node,child)
                children_dist = children_dist + child_dist
            B1[i] = children_weight1
            sum_dist = children_dist + p_dist
            A1[i][ref_table[node.name]] = sum_dist
            i = i+1
    # update the integer 0 to vector with 0s
    for x in range(len(B1)):
        if (type(B1[x]) == int):
            B1[x] = np.array([0 for i in range(landmark_len(t))])
    result = np.linalg.solve(A1,B1)
    t_result = t.copy("deepcopy")
    dir = 'new_landmark/'
    failsafe_mkdir(dir)
    for node in t_result.traverse("levelorder"):
        if node.is_leaf() == False:
            node.add_features(weight=result[ref_table.get(node.name)])
        internal_weight = np.array(node.weight).reshape(int(landmark_len(t)/3),3)
        np.savetxt(dir+node.name+'.txt',internal_weight,fmt='%.14f')
    return t_result

# update branch length to reflect norphological change
def update_branch(t):
    t_result = t.copy("deepcopy")
    for node in t_result.traverse("levelorder"):
        if node.up != None:
            node_dist = scipy.spatial.distance.euclidean(node.weight,node.up.weight)
            node.dist = float(round(node_dist,5))
    return t_result

# helper function used in weight solver function
def failsafe_mkdir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        #print("Landmark file already created")
        shutil.rmtree(dir)
        os.mkdir(dir)

# verify the internal node weight
def result_verification(t):
    #print("Veriying the internal nodes weight...")
    for node in t.traverse("levelorder"):
        if node.is_leaf() == False:
            print(node.name)
            sum_dist = 0.0
            p_dist = 0.0
            parent = 0.0
            children_dist = 0.0
            children_weight = 0.0
        # if the node is not root
            if node.up != None:
                parent = node.up.weight
                p_dist = 1/t.get_distance(node,node.up)
            for child in node.get_children():
                child_dist = 1/t.get_distance(node,child)
                children_dist = children_dist + child_dist
                children_weight = children_weight + child_dist*child.weight
            sum_dist = children_dist + p_dist
            left = sum_dist*node.weight
            right = children_weight + p_dist*parent
            check = scipy.spatial.distance.euclidean(left, right)
            print("%0.10f" %check)

# load the foreign fossil landmark
def read_foreign(foreign):
    try:
        data = np.loadtxt(foreign, delimiter=' ')
    except ValueError:
        data = np.loadtxt(foreign, delimiter=',')
    foreign_fossil = data.flatten()
    return foreign_fossil

# calculate euclidean distance from foreign fossil to any edge
# if the projection goes out of the edge, choose the parent/child node
# that is closer to the fossil as projection
def euclidean_edge_segment(f,tree):
    # euclidean distance from the fossil to all edges
    euc_distance = {}
    # euclidean distance from the fossil to the parent node on edge
    euc_parent = {}
    # euclidean distance from the fossil to the child node on edge
    euc_child={}
    # mark edge with min/max distance to the fossil
    min_distance = {}
    max_distance = {}
    # percentage of distance from child to p_base
    base_distance_ratio = {}
    for node in tree.traverse("levelorder"):
        if node.is_leaf() == False:
            p0 = node.weight
            for child in node.get_children():
                d = 0
                # add the Euclidean distance to the child end of an edge
                edge = child.name
                p1 = child.weight
                v = p1-p0
                w = f-p0
                t = np.dot(w,v)/np.dot(v,v)
                if (t<=0):
                    #d = scipy.spatial.distance.euclidean(f,p0)
                    d = distance(f,p0)
                    p_base = p0
                elif (t>=1):
                    #d = scipy.spatial.distance.euclidean(f,p1)
                    d = distance(f,p1)
                    p_base = p1
                else:
                    p_base = p0 + t*v
                    #d = scipy.spatial.distance.euclidean(f,p_base)
                    d = distance(f,p_base)
                euc_distance[edge] = float(round(d,5))
                #euc_parent[edge] = float(round(scipy.spatial.distance.euclidean(f,child.weight),5))
                euc_parent[edge] = float(round(distance(f,child.weight),5))
                #euc_child[edge] = float(round(scipy.spatial.distance.euclidean(f,node.weight),5))
                euc_child[edge] = float(round(distance(f,node.weight),5))
                #base_distance_ratio[edge] = float(round(scipy.spatial.distance.euclidean(p_base,p1)/scipy.spatial.distance.euclidean(p0,p1),5))
                base_distance_ratio[edge] = float(round(distance(p_base,p1)/distance(p0,p1),5))
                # Heron's formula to check the first dot product method
                #s = float(scipy.spatial.distance.euclidean(f,p0)+scipy.spatial.distance.euclidean(f,p1)+scipy.spatial.distance.euclidean(p0,p1))/2
                #area = math.sqrt(s*(s-scipy.spatial.distance.euclidean(f,p0))*(s-scipy.spatial.distance.euclidean(f,p1))*(s-scipy.spatial.distance.euclidean(p0,p1)))
                #d2 = float(round(2*area/scipy.spatial.distance.euclidean(p0,p1),8))
                #print(round(scipy.spatial.distance.euclidean(f,p0),8))
                #print("distance difference",round(float(d-d2),8))
                #print("vector",d)
                #print("Heron",d2)
                min_distance[edge] = 0
                max_distance[edge] = 0
    shortest = min(euc_distance, key=euc_distance.get)
    longest_e = max(euc_distance, key=euc_distance.get)
    longest_p = max(euc_parent, key=euc_parent.get)
    longest_c = max(euc_child, key=euc_child.get)
    longest = max(longest_e,longest_p,longest_c)
    min_distance[shortest] = 1
    max_distance[longest] = 1
    for node in tree.traverse("levelorder"):
        node.add_features(euc_dist=euc_distance.get(node.name, "none"))
        node.add_features(euc_parent=euc_parent.get(node.name, "none"))
        node.add_features(euc_child=euc_child.get(node.name, "none"))
        node.add_features(min_dist=min_distance.get(node.name, "none"))
        node.add_features(max_dist=max_distance.get(node.name, "none"))
        node.add_features(dist_ratio=base_distance_ratio.get(node.name, "none"))
    # sort the distance from fossil to edge from shortest to longest
    #d_sorted_by_value = OrderedDict(sorted(euc_distance.items(), key=lambda x: x[1]))
    #for k, v in d_sorted_by_value.items():
    #	print("%s: %s" % (k, v))
    return tree

####################
# Refitting the foreign fossil back to original tree
# level 1, find the projection s of foreign fossil f on an edge
def fit_fossil_level_1(f,tree):
    t1 = tree.copy("deepcopy")
    p1 = t1.search_nodes(min_dist=1)[0]
    p0 = p1.up
    #dist_original = tree.get_distance(p1, p0)
    v = p1.weight - p0.weight
    w = f - p0.weight
    t = np.dot(w,v)/np.dot(v,v)
    # calculate the projection of foreign fossil onto the edge
    if(t <= 0):
        s_weight = p0.weight
        fr = p0.add_child(name="foreign")
        #d = scipy.spatial.distance.euclidean(f,p0.weight)
        d = distance(f,p0.weight)
        # update edge length from foreign node to p0
        fr.dist = float(round(d,5))
        #print("t<=0")
    elif (t >= 1):
        s_weight = p1.weight
        fr = p1.add_child(name="foreign")
        #d = scipy.spatial.distance.euclidean(f,p1.weight)
        d = distance(f,p1.weight)
        # update edge length from foreign node to p1
        fr.dist = float(round(d,5))
        #print("t>=1",t)
    else:
        s_weight = p0.weight+t*v
        #d = scipy.spatial.distance.euclidean(f,s_weight)
        d = distance(f,s_weight)
        removed1 = p1.detach()
        s = p0.add_child(name="s")
        fr = s.add_child(name="foreign")
        removed1 = p1.detach()
        s.add_child(removed1)
        s.add_features(weight=s_weight)
        fr.add_features(weight=f)
        #p1.dist = float(round(scipy.spatial.distance.euclidean(p1.weight,s_weight),5))
        p1.dist = float(round(distance(p1.weight,s_weight),5))
        fr.dist = float(round(d,5))
        #s.dist = float(round(scipy.spatial.distance.euclidean(s_weight,p0.weight),5))
        s.dist = float(round(distance(s_weight,p0.weight),5))
    return t1

# level 2 refit
def fit_fossil_level_2(f,tree):
    t2 = tree.copy("deepcopy")
    p1 = t2.search_nodes(min_dist=1)[0]
    p0 = p1.up
    s_weight = 1/3*(p1.weight+p0.weight+f)
    removed1 = p1.detach()
    s = p0.add_child(name="s")
    fr = s.add_child(name="foreign")
    s.add_child(removed1)
    fr.add_features(weight=f)
    s.add_features(weight=s_weight)
    #p1.dist = float(round(scipy.spatial.distance.euclidean(p1.weight,s_weight),5))
    #fr.dist = float(round(scipy.spatial.distance.euclidean(f,s_weight),5))
    #s.dist = float(round(scipy.spatial.distance.euclidean(s_weight,p0.weight),5))
    p1.dist = float(round(distance(p1.weight,s_weight),5))
    fr.dist = float(round(distance(f,s_weight),5))
    s.dist = float(round(distance(s_weight,p0.weight),5))
    return t2

def node_distance(t):
	n_distance = {}
	for node in t.traverse("postorder"):
		if node.is_leaf() == True:
			l_distance = {}
			for leaf in t.traverse("postorder"):
				if leaf.is_leaf() == True:
					l_distance[leaf.name] = t.get_distance(leaf,node)
			n_distance[node.name] = l_distance
	return n_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='+', help='input file (in order): original Newick tree (required), landmark coordinates (optional), fossil for insertion (optional)')
    parser.add_argument('-n', action='store_true', help='Compute the internal nodes and branch lengths to create a new tree, input original tree and landmark coordiantes')
    parser.add_argument('-v', action='store_true', help='Verify the internal nodes weight, input original tree and landmark coordiantesinput original tree and landmark coordiantes')
    parser.add_argument('-f', action='store_true', help='Fit a fossil to given tree, input original tree, landmark coordiantes, fossil for insertion')
    args = parser.parse_args()
    in_file = args.file[0]
    landmark = args.file[1]

    # load the original tree
    original_tree = read_tree(in_file)

    # add lanmarks to fossils
    out_tree = add_landmarkWeight(original_tree,landmark)

    # solve for internal fossil weight
    t_result_1 = weight_solver(out_tree)
    t_result = update_branch(t_result_1)

    if args.n:
        # write the new tree to a newick file
        t_result.write(format=1, outfile="new_tree.nw")
        print("Computing new phylogenetic tree")
    elif args.v:
        # verify the result
        print("Veriifying internal nodes weights")
        result_verification(t_result_1)
    elif args.f:
        print("Inserting fossil to tree\n")
        t_result.write(format=1, outfile="new_tree.nw")
        foreign = args.file[2]
        # load the fossil to be inserted in the new tree
        foreign_fossil = read_foreign(foreign)

        # update the distances calculated from fossil to tree
        t_euc = euclidean_edge_segment(foreign_fossil,t_result)

        # create a tree in Newick format with necessary attributes for d3 visualization
        t_euc.write(format=1,features=["euc_dist","min_dist","max_dist","dist_ratio", "euc_parent","euc_child"], outfile="new_tree_d3_2.nw")

        print("Original tree:")
        print(t_euc.get_ascii(show_internal=True))
        # level 1 refit
        t_refit_1 = fit_fossil_level_1(foreign_fossil,t_euc)
        t_refit_1.write(format=1, outfile=str(get_name(foreign)+'_level1.nw'))
        print("\nLevel 1 fitting of %s" %get_name(foreign))
        print(t_refit_1.get_ascii(show_internal=True))

        # level 2 refit
        t_refit_2 = fit_fossil_level_2(foreign_fossil,t_euc)
        t_refit_2.write(format=1, outfile=str(get_name(foreign)+'_level2.nw'))
        print("\nLevel 2 fitting of %s" %get_name(foreign))
        print(t_refit_2.get_ascii(show_internal=True))
        
    else:
        print("Please provide valid inputs to the program, enter -h for help")

if __name__== "__main__":
  main()



