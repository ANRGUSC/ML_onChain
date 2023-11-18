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
                data.append([name, deployment_gas, upload_test_gas, weight_biases_gas, classify_gas])

    columns = ['name', 'deployment_gas', 'upload_test_gas', 'weight_biases_gas', 'classify_gas']
    return pd.DataFrame(data, columns=columns)


# Replace 'your_file.txt' with the path to your text file
file_path = '../results/Onchain_accuracy'
df = read_data(file_path)
print(df)

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

# Plotting graphs for each column
plot_graph(df, 'deployment_gas', 'Deployment Gas vs Model')
plot_graph(df, 'upload_test_gas', 'Test Data Upload Gas vs Model')
plot_graph(df, 'weight_biases_gas', 'Weights and Biases Upload Gas vs Model')
plot_graph(df, 'classify_gas', 'Classify Gas vs Model')
