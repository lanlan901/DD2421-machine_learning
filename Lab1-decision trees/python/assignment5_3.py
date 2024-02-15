import monkdata as m
import dtree as d
import drawtree_qt5

# 处理MONK-3数据集
tree3 = d.buildTree(m.monk3, m.attributes)
train_error3 = 1 - d.check(tree3, m.monk3)
test_error3 = 1 - d.check(tree3, m.monk3test)

# 打印性能信息
print("MONK-3 Dataset:")
print(f"Performance on the test data: {d.check(tree3, m.monk3test)}")
print(f"Training Set Error: {train_error3}")
print(f"Test Set Error: {test_error3}")

# 绘制决策树
drawtree_qt5.drawTree(tree3)