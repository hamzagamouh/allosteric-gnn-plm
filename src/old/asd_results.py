import re
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    model_sections = {'GNN': [], 'PLM': []}
    current_model = None

    for line in lines:
        if 'GNN -' in line:
            current_model = 'GNN'
        elif 'PLM Baseline' in line:
            current_model = 'PLM'
        elif current_model:
            model_sections[current_model].append(line.strip())

    return model_sections

def extract_metrics(log_lines, model_type='GNN'):
    epochs = []
    train_auc = []
    val_auc = []
    train_mcc = []
    val_mcc = []

    for line in log_lines:
        epoch_match = re.match(r"Epoch (\d+)", line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        if model_type == 'GNN':
            auc_mcc_match = re.match(
                r"Train AUC ([0-9.]+) Train MCC ([\-0-9.]+)", line)
            val_match = re.match(
                r"Val AUC ([0-9.]+) Val MCC ([\-0-9.]+)", line)
        elif model_type == 'PLM':
            auc_mcc_match = re.match(
                r"Train AUC ([0-9.]+) Train MCC ([\-0-9.]+)", line)
            val_match = re.match(
                r"Val AUC ([0-9.]+) Val MCC ([\-0-9.]+)", line)
        else:
            continue

        if auc_mcc_match:
            if model_type == 'GNN':
                train_auc.append(float(auc_mcc_match.group(1)))
                train_mcc.append(float(auc_mcc_match.group(2)))
            elif model_type == 'PLM':
                train_auc.append(float(auc_mcc_match.group(1)))
                train_mcc.append(float(auc_mcc_match.group(2)))

        if val_match:
            if model_type == 'GNN':
                val_auc.append(float(val_match.group(1)))
                val_mcc.append(float(val_match.group(2)))
            elif model_type == 'PLM':
                val_auc.append(float(val_match.group(1)))
                val_mcc.append(float(val_match.group(2)))

    return {
        'epochs': epochs,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'train_mcc': train_mcc,
        'val_mcc': val_mcc
    }

def plot_metrics(data_gnn, data_plm):
    plt.figure(figsize=(16, 10))

    # GNN AUC
    plt.subplot(2, 2, 1)
    plt.plot(data_gnn['epochs'], data_gnn['train_auc'], label='Train AUC', color='blue')
    plt.plot(data_gnn['epochs'], data_gnn['val_auc'], label='Val AUC', color='cyan')
    plt.title('GNN AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    # GNN MCC
    plt.subplot(2, 2, 2)
    plt.plot(data_gnn['epochs'], data_gnn['train_mcc'], label='Train MCC', color='blue')
    plt.plot(data_gnn['epochs'], data_gnn['val_mcc'], label='Val MCC', color='cyan')
    plt.title('GNN MCC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()
    plt.grid(True)

    # PLM AUC
    plt.subplot(2, 2, 3)
    plt.plot(data_plm['epochs'], data_plm['train_auc'], label='Train AUC', color='red')
    plt.plot(data_plm['epochs'], data_plm['val_auc'], label='Val AUC', color='orange')
    plt.title('PLM AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    # PLM MCC
    plt.subplot(2, 2, 4)
    plt.plot(data_plm['epochs'], data_plm['train_mcc'], label='Train MCC', color='red')
    plt.plot(data_plm['epochs'], data_plm['val_mcc'], label='Val MCC', color='orange')
    plt.title('PLM MCC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("asd_results.png")
    plt.show()

if __name__ == "__main__":
    filepath = "asd_results.txt"  # <-- Change this to your actual file path
    logs = parse_log_file(filepath)

    gnn_data = extract_metrics(logs['GNN'], model_type='GNN')
    plm_data = extract_metrics(logs['PLM'], model_type='PLM')
    # print(gnn_data)
    # print(plm_data)
    plot_metrics(gnn_data, plm_data)
