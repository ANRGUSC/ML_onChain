import pandas as pd
import matplotlib.pyplot as plt


# Function to read data from the text file and create a DataFrame
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



# Replace 'your_file.txt' with the path to your text file
file_path = '../results/Onchain_accuracy'
df = read_data(file_path)
df['total_gas_cost'] = df['deployment_gas'] + df['upload_test_gas'] + df['weight_biases_gas'] + df['classify_gas']

# Read local accuracy data
local_file_path = '../results/Local_accuracy'
local_accuracies = read_local_accuracy(local_file_path)
df['local_accuracy'] = df['name'].map(local_accuracies)


# Function to plot a graph
def plot_graph(df, column, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df['name'], df[column], marker='o')
    plt.title(title)
    plt.xlabel('Model Name')
    plt.ylabel(column.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    plt.savefig("../results/visualization/"+column+".png")


def plot_accuracy_comparison(df, onchain_column, local_column, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df['name'], df[local_column], marker='o', color='green', label='Local Accuracy', linewidth=6,markersize = 10)
    plt.plot(df['name'], df[onchain_column], marker='o', color='orange', label='On-chain Accuracy')
    plt.xlabel('Model Name')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("../results/visualization/" + 'accuracy' + ".png")


# Plotting graphs for each column
plot_graph(df, 'deployment_gas', 'Deployment Gas vs Model')
plot_graph(df, 'upload_test_gas', 'Test Data Upload Gas vs Model')
plot_graph(df, 'weight_biases_gas', 'Weights and Biases Upload Gas vs Model')
plot_graph(df, 'classify_gas', 'Classify Gas vs Model')

# Plotting the new graph for total gas cost
plot_graph(df, 'total_gas_cost', 'Total Gas Cost vs Model')

# Plotting the comparison graph
plot_accuracy_comparison(df, 'accuracy', 'local_accuracy', 'On-chain vs Local Accuracy Comparison')
