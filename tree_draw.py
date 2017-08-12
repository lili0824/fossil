import sys
import argparse
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

ts = TreeStyle()
ts.show_leaf_name = False
ts.show_branch_length = True

def my_layout(node):
    F = TextFace(node.name, tight_text=True)
    if node.is_leaf():
        add_face_to_node(F, node, column=0, position="branch-right")
    else:
        add_face_to_node(F, node, column=0, position="branch-bottom")
ts.layout_fn = my_layout

def read_tree(in_file):
    with open(in_file,'r') as file_ptr:
        in_tree = file_ptr.read()
    out_tree = Tree(in_tree,format=1)
    return out_tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='+', help='input file: Newick file to be drawn')
    args = parser.parse_args()
    in_file = args.file[0]

    t = read_tree(in_file)
    t.show(tree_style=ts)
        
if __name__== "__main__":
  main()












