import monkdata as m
import dtree as d
import random
import matplotlib.pyplot as plt

num_runs = 200


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune_tree(train_data, validation_data):
    tree = d.buildTree(train_data, m.attributes)
    best_tree = tree
    best_performance = d.check(best_tree, validation_data)
    while True:
        improved = False
        for pruned_tree in d.allPruned(tree):
            performance = d.check(pruned_tree, validation_data)
            if performance > best_performance:
                best_tree = pruned_tree
                best_performance = performance
                improved = True
        if not improved:
            break
    return best_tree


# fractions to test
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

results_monk1 = {fraction: [] for fraction in fractions}
results_monk3 = {fraction: [] for fraction in fractions}

for fraction in fractions:
    for _ in range(num_runs):
        # MONK-1
        monk1train, monk1val = partition(m.monk1, fraction)
        pruned_tree_monk1 = prune_tree(monk1train, monk1val)
        test_error_monk1 = 1 - d.check(pruned_tree_monk1, m.monk1test)
        results_monk1[fraction].append(test_error_monk1)

        # MONK-3
        monk3train, monk3val = partition(m.monk3, fraction)
        pruned_tree_monk3 = prune_tree(monk3train, monk3val)
        test_error_monk3 = 1 - d.check(pruned_tree_monk3, m.monk3test)
        results_monk3[fraction].append(test_error_monk3)

# Calculating mean and standard deviation
mean_errors_monk1 = {fraction: sum(errors) / len(errors) for fraction, errors in results_monk1.items()}
std_dev_monk1 = {fraction: (sum((x - mean_errors_monk1[fraction]) ** 2 for x in errors) / len(errors)) ** 0.5 for
                 fraction, errors in results_monk1.items()}

mean_errors_monk3 = {fraction: sum(errors) / len(errors) for fraction, errors in results_monk3.items()}
std_dev_monk3 = {fraction: (sum((x - mean_errors_monk3[fraction]) ** 2 for x in errors) / len(errors)) ** 0.5 for
                 fraction, errors in results_monk3.items()}

# Plot
plt.errorbar(list(mean_errors_monk1.keys()), list(mean_errors_monk1.values()), yerr=list(std_dev_monk1.values()),
             fmt='o', label='MONK-1')
plt.errorbar(list(mean_errors_monk3.keys()), list(mean_errors_monk3.values()), yerr=list(std_dev_monk3.values()),
             fmt='o', label='MONK-3')
plt.xlabel('Fraction')
plt.ylabel('Test Error')
plt.title('Test Error vs Fraction for MONK-1 and MONK-3')
plt.legend()
plt.show()
