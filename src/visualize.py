import pandas as pd
import matplotlib.pyplot as plt


# Function to read data from the Onchain_accuracy file and create a DataFrame
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Accuracy'):
                accuracy = float(line.split(':')[1].strip().replace('%', ''))
            elif line.startswith('Name'):
                name = line.split(':')[1].strip()
            elif line.startswith('Deployment Gas'):
                deployment_gas = int(line.split(':')[1].strip())
            elif line.startswith('Test data upload gas'):
                upload_test_gas = int(line.split(':')[1].strip())
            elif line.startswith('Weights and biases upload gas'):
                weight_biases_gas = int(line.split(':')[1].strip())
            elif line.startswith('Classify gas'):
                classify_gas = int(line.split(':')[1].strip())
                data.append([accuracy, name, deployment_gas, upload_test_gas, weight_biases_gas, classify_gas])

    columns = ['accuracy', 'name', 'deployment_gas', 'upload_test_gas', 'weight_biases_gas', 'classify_gas']
    return pd.DataFrame(data, columns=columns)


# Function to plot the Deployment Cost graph as a line chart
def plot_deployment_cost_line(df):
    plt.figure()
    plt.plot(df['name'], df['weight_biases_gas'], marker='o', label='Upload Weights and Biases Cost')
    plt.plot(df['name'], df['deployment_gas'], marker='o', label='Model Deployment Cost')
    plt.ylabel('Gas Cost')
    plt.title('Deployment Cost')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("../results/Visualization/deployment_cost_line.png")
    plt.show()


# Function to plot the Inference Cost graph
def plot_inference_cost(df):
    plt.figure()
    plt.plot(df['name'], df['upload_test_gas'], marker='o', label='Model Uploading Cost')
    plt.plot(df['name'], df['classify_gas'], marker='o', label='Classification Cost')
    plt.ylabel('Gas Cost')
    plt.title('Inference Cost')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("../results/Visualization/inference_cost.png")
    plt.show()


# Reading data from the provided file
local_accuracy_file_path = '../results/Local_accuracy'
onchain_file_path = '../results/Onchain_accuracy'
df = read_data(onchain_file_path)
df['name'] = ['1L1N', '2L1N', '2L2N', '2L3N', '2L4N', '3L1N', '3L2N', '3L3N', '3L4N']
# Plotting the specific graphs
plot_deployment_cost_line(df)
plot_inference_cost(df)


# -----------------------------------------------------------------
# plot the accuracy comparison
# -----------------------------------------------------------------
# Function to read local accuracy data
def read_local_accuracy(file_path):
    local_accuracy_data = {}
    with open(file_path, 'r') as file:
        while True:
            name_line = file.readline().strip()
            if not name_line:
                break  # End of file or empty line
            accuracy_line = file.readline().strip()
            if not accuracy_line:
                break  # Missing accuracy line, should not happen in well-formed file
            try:
                accuracy = float(accuracy_line.split(':')[1].strip().replace('%', ''))
                local_accuracy_data[name_line] = accuracy
            except ValueError:
                print(f"Invalid format for accuracy in line: {accuracy_line}")
                break  # Break on unexpected format
            # Skip the empty line after each accuracy
            file.readline()
    return local_accuracy_data


# Function to plot accuracy comparison

def plot_accuracy_comparison(df, onchain_column, local_column, title):
    # Set the width of the bars
    bar_width = 0.35
    # Set the positions of the bars
    index = range(len(df))
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(index, df[local_column], bar_width, label='Local Accuracy')
    plt.bar([i + bar_width for i in index], df[onchain_column], bar_width, label='On-chain Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks([i + bar_width / 2 for i in index], df['name'], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig("../results/visualization/accuracy_comparison_bar.png")
    plt.show()

local_accuracies = read_local_accuracy(local_accuracy_file_path)
df['local_accuracy'] = df['name'].map(local_accuracies)

plot_accuracy_comparison(df, 'accuracy', 'local_accuracy', 'On-chain vs Local Accuracy Comparison')

# -----------------------------------------------------------------
# plot the gas cost comparison
# -----------------------------------------------------------------
# Dummy values for gas costs
sol_builtin_gas = {'add': 170, 'mul': 230, 'div': 274}
prbmath_gas = {'add': 382, 'mul': 656, 'div': 617}

# Labels for the bar chart
labels = sol_builtin_gas.keys()

# Values for each group of bars
sol_values = [sol_builtin_gas[label] for label in labels]
prb_values = [prbmath_gas[label] for label in labels]

# Set up the bar chart
x = range(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, sol_values, width, label='Solidity Built-in')
rects2 = ax.bar([p + width for p in x], prb_values, width, label='PRBMath')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Operation')
ax.set_ylabel('Gas Cost')
ax.set_title('Comparison of Gas Costs: Solidity Built-in vs PRBMath')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(labels)
ax.legend()


# Function to add labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
plt.savefig("../results/visualization/" + 'gas_cost_comparison' + ".png")
plt.show()
