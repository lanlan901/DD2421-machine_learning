import monkdata as m
import dtree as d
import drawtree_qt5

# 处理MONK-2数据集
tree2 = d.buildTree(m.monk2, m.attributes)
train_error2 = 1 - d.check(tree2, m.monk2)
test_error2 = 1 - d.check(tree2, m.monk2test)

# 打印性能信息
print("MONK-2 Dataset:")
print(f"Performance on the test data: {d.check(tree2, m.monk2test)}")
print(f"Training Set Error: {train_error2}")
print(f"Test Set Error: {test_error2}")

# 绘制决策树
drawtree_qt5.drawTree(tree2)
