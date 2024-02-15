import monkdata as m
import dtree as d
import drawtree_qt5

# 处理MONK-1数据集
tree1 = d.buildTree(m.monk1, m.attributes)
train_error1 = 1 - d.check(tree1, m.monk1)
test_error1 = 1 - d.check(tree1, m.monk1test)

# 打印性能信息
print("MONK-1 Dataset:")
print(f"Performance on the test data: {d.check(tree1, m.monk1test)}")
print(f"Training Set Error: {train_error1}")
print(f"Test Set Error: {test_error1}")

# 绘制决策树
drawtree_qt5.drawTree(tree1)