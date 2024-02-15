import monkdata as m
import dtree as d
import drawtree_qt5

root_node = d.bestAttribute(m.monk3, m.attributes)
for value in root_node.values:
    subset = d.select(m.monk3, root_node, value)
    print(f"Subset for Attribute {root_node} = {value}")
    for attr in m.attributes:
        if attr != root_node:
            gain = d.averageGain(subset, attr)
            print(f"  Information Gain for {attr}: {gain}")
    majority_class = d.mostCommon(subset)
    print(f"  Majority class for this subset: {'Positive' if majority_class else 'Negative'}")
    print()

#ID3
tree = d.buildTree(m.monk3, m.attributes, 2)
drawtree_qt5.drawTree(tree)