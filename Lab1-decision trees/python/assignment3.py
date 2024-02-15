import monkdata as m
import dtree

datasets = [m.monk1, m.monk2, m.monk3]
for i, dataset in enumerate(datasets, start=1):
    print(f"MONK-{i} Dataset:")
    for j, attribute in enumerate(m.attributes):
        gain = dtree.averageGain(dataset, attribute)
        print(f"Attribute A{j+1}: Information Gain = {gain}")
    print()
