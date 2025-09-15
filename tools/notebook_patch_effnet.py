import json
import os

NB_PATH = os.path.join('effnet_multi_classification', 'effnet_model_training.ipynb')

def load_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_nb(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def make_run_experiment_source():
    code = r'''
# Helper function for the sampler
def create_sampler(train_ds):
    class_counts = np.bincount([label for label in train_ds.labels if label != -1], minlength=num_classes)
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = np.array([class_weights[t] if t != -1 else 0 for t in train_ds.labels])
    return WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights))

def run_experiment(params):
    """
    Runs a full training and validation experiment for a given set of hyperparameters.
    Adds LR scheduling, early stopping, per-run dirs, and returns model_path.
    """
    print("\n" + "="*50)
    print(f"Params: {params}")
    print("="*50)

    # Reproducibility
    try:
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as _e:
        pass

    start_time = time.time()

    # Defaults
    EPOCHS = int(params.get('epochs', 10))
    es_patience = int(params.get('early_stopping_patience', 5))
    rlrop_patience = int(params.get('reduce_lr_patience', 2))
    rlrop_factor = float(params.get('reduce_lr_factor', 0.5))

    # --- 1. Setup Transforms, Datasets, and DataLoaders ---
    transform = transforms.Compose([
        transforms.Resize((params['image_size'], params['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MalariaDataset(train_json_path, image_path, transform=transform, category_map=category_map)
    test_ds = MalariaDataset(test_json_path, image_path, transform=transform, category_map=category_map)

    sampler = None
    shuffle = True
    if params['sampling'] == 'oversample':
        print("Applying weighted oversampling...")
        sampler = create_sampler(train_ds)
        shuffle = False

    pin_mem = True if str(DEVICE) == 'cuda' else False
    train_loader = DataLoader(
        train_ds, batch_size=params['batch_size'], shuffle=shuffle, sampler=sampler,
        collate_fn=custom_collate_fn, num_workers=int(params.get('num_workers', 2)), pin_memory=pin_mem
    )
    val_loader = DataLoader(
        test_ds, batch_size=params['batch_size'], shuffle=False,
        collate_fn=custom_collate_fn, num_workers=int(params.get('num_workers', 2)), pin_memory=pin_mem
    )

    # --- 2. Initialize Model and Optimizer ---
    model = EfficientNetDetector(num_classes=num_classes).to(DEVICE)
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=rlrop_patience, factor=rlrop_factor, verbose=True
    )

    # Prepare run dir and history
    sampling_str = params['sampling'] if params['sampling'] is not None else 'none'
    run_name = (f"run_lr-{params['lr']}_optim-{params['optimizer']}_bs-{params['batch_size']}"
                f"_img-{params['image_size']}_sampling-{sampling_str}")
    run_dir = os.path.join(models_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # --- 3. Training Loop ---
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, DEVICE, epoch, num_classes)
        val_loss, val_acc = validate_model(model, val_loader, DEVICE, category_map)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Save best by accuracy (primary metric)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"New best model! Val Accuracy: {best_val_accuracy:.2f}%")
            model_filename = (f"model_lr-{params['lr']}_optim-{params['optimizer']}_bs-{params['batch_size']}_sampling-{sampling_str}.pth")
            save_path = os.path.join(models_dir, model_filename)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_accuracy': best_val_accuracy,
                'params': params
            }, save_path)
            best_model_path = save_path
            print(f"Model saved to {save_path}")

        # Early stopping on val_loss
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Optional: write intermediate history
        try:
            with open(os.path.join(run_dir, 'history_partial.json'), 'w') as hf:
                json.dump(history, hf, indent=2)
        except Exception:
            pass

    # Save final history
    try:
        with open(os.path.join(run_dir, 'history.json'), 'w') as hf:
            json.dump(history, hf, indent=2)
    except Exception:
        pass

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished experiment. Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Total time: {elapsed_time / 60:.2f} minutes")

    return {
        'params': params,
        'best_accuracy': best_val_accuracy,
        'training_time_minutes': elapsed_time / 60,
        'history': history,
        'model_path': best_model_path if best_model_path is not None else '',
        'run_dir': run_dir
    }
'''
    return [line + "\n" for line in code.splitlines()]

def make_final_eval_source():
    code = r'''
# --- 1. Analyze the Grid Search Results ---
print("--- Analyzing Grid Search Results ---")
results_filepath = os.path.join(models_dir, 'grid_search_results.json')
results_df = pd.read_json(results_filepath)
params_df = pd.json_normalize(results_df['params'])
results_df = pd.concat([results_df.drop('params', axis=1), params_df], axis=1)
# Backfill model_path if missing
if 'model_path' not in results_df.columns or results_df['model_path'].isna().any():
    def _mk_path(r):
        samp = r.get('sampling', 'none')
        samp = samp if samp is not None else 'none'
        fname = f"model_lr-{r['lr']}_optim-{r['optimizer']}_bs-{r['batch_size']}_sampling-{samp}.pth"
        return os.path.join(models_dir, fname)
    results_df['model_path'] = results_df.apply(_mk_path, axis=1)

results_df_sorted = results_df.sort_values(by='best_accuracy', ascending=False)
print("\n--- Grid Search Results Summary ---")
print(results_df_sorted.head())

# --- 2. Start Final Evaluation on the Best Model ---
print(f"\n{'='*50}\n--- Starting Final Evaluation on the Best Model ---\n{'='*50}")
best_run = results_df_sorted.iloc[0]
best_model_path = best_run['model_path']
best_params = best_run.to_dict()

print(f"\nLoading best model from: {best_model_path}")
print(f"Best model parameters found:\n{json.dumps(best_params, indent=4)}")

checkpoint = torch.load(best_model_path, map_m
ap=DEVICE)
eval_model = EfficientNetDetector(num_classes=num_classes).to(DEVICE)
eval_model.load_state_dict(checkpoint['model_state_dict'])
eval_model.eval()

final_test_loader = DataLoader(test_ds, batch_size=int(best_params['batch_size']), shuffle=False, collate_fn=custom_collate_fn)
y_true, y_pred = validate_model(eval_model, final_test_loader, DEVICE, category_map, return_preds=True)

# --- 3. Save Final Predictions and Generate Report ---
final_results_filepath = os.path.join(models_dir, 'best_model_final_predictions.csv')
true_cols = [f'true_{name.replace(" ", "_")}' for name in class_names]
pred_cols = [f'pred_{name.replace(" ", "_")}' for name in class_names]
final_results_df = pd.DataFrame(np.hstack([y_true, y_pred]), columns=true_cols + pred_cols)
final_results_df.to_csv(final_results_filepath, index_label='image_index')
print(f"\nFinal predictions saved to: {final_results_filepath}")

print("\n--- Classification Report ---")
report = classification_report(y_true, y_pred, target_names=class_names, labels=list(range(len(class_names))), zero_division=0)
print(report)

# --- 4. Grad-CAM: define helper and save overlays for a few samples ---
import torch.nn.functional as F
class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.fh = target_module.register_forward_hook(self._forward_hook)
        self.bh = target_module.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    def generate(self, input_tensor, target_index):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_index is None:
            target_index = torch.argmax(logits, dim=1)
        score = logits.gather(1, target_index.view(-1,1)).sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
    def close(self):
        self.fh.remove(); self.bh.remove()

def _find_last_conv(m):
    last = None
    for mm in m.modules():
        if isinstance(mm, nn.Conv2d):
            last = mm
    return last

def _denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1,3,1,1)
    return (img_tensor * std) + mean

target_layer = _find_last_conv(eval_model.backbone.features)
gc = GradCAM(eval_model, target_layer)
gradcam_dir = os.path.join(models_dir, 'gradcam_best')
os.makedirs(gradcam_dir, exist_ok=True)
# take one batch from test loader
images_batch, targets_list = next(iter(final_test_loader))
images_batch = images_batch.to(DEVICE)
with torch.no_grad():
    logits = eval_model(images_batch)
probs = torch.sigmoid(logits)
top_idx = torch.argmax(logits, dim=1)
cams = gc.generate(images_batch, top_idx)
# Save first 8 overlays
import matplotlib.pyplot as plt
for i in range(min(8, images_batch.size(0))):
    img = _denorm(images_batch[i:i+1]).clamp(0,1).cpu().squeeze(0).permute(1,2,0).numpy()
    cam = F.interpolate(cams[i:i+1], size=(img.shape[0], img.shape[1]), mode='bilinear', align_corners=False).cpu().squeeze().numpy()
    plt.figure(figsize=(4,4)); plt.axis('off')
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.35)
    out_path = os.path.join(gradcam_dir, f'gradcam_{i}.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
print(f"Saved Grad-CAM samples to: {gradcam_dir}")
'''
    # ensure lines end with \n
    return [line + "\n" for line in code.splitlines()]

def patch_notebook(nb):
    updated_run_exp = False
    updated_final_eval = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            src_txt = ''.join(cell.get('source', []))
            if 'def run_experiment(params):' in src_txt:
                # Replace entire cell source with our improved function + helper
                cell['source'] = make_run_experiment_source()
                updated_run_exp = True
            elif '# --- 1. Analyze the Grid Search Results ---' in src_txt:
                # Fix any previously inserted bad map_location token first by direct replace
                fixed = src_txt.replace('map_m\nap=DEVICE', 'map_location=DEVICE')
                if fixed != src_txt:
                    cell['source'] = [l + "\n" for l in fixed.splitlines()]
                else:
                    cell['source'] = make_final_eval_source()
                updated_final_eval = True
    return updated_run_exp, updated_final_eval

def main():
    nb = load_nb(NB_PATH)
    updated_run_exp, updated_final_eval = patch_notebook(nb)
    save_nb(NB_PATH, nb)
    print(f"Updated run_experiment: {updated_run_exp}, final_eval: {updated_final_eval}")

if __name__ == '__main__':
    main()
