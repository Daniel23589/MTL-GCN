import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
#from randaugment import RandAugment
from torchvision.transforms import ColorJitter, RandomHorizontalFlip
import xml.dom.minidom


class mafood(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # data_split = train / val / test
        self.root = root
        self.classnames = ['ff_bread', 'ff_dumpling', 'ff_egg', 'ff_fried_food', 'ff_meat', 'ff_noodles/pasta', 'ff_rice', 'ff_seafood', 'ff_soup', 'ff_vegetable']
        
        # Convertimos a lowercase para asegurar consistencia
        self.data_split = data_split.lower()

        # Base path del dataset (ajusta si es necesario)
        dataset_base_path = '/media/HDD_4TB_2/daniel/gnn_test/3.MLR-GCN/MLR-GCN_MTL_6/food_datasets/MAFood121/'

        # Carga de etiquetas multi-label (ingredientes)
        self.labels_lab = np.load(os.path.join(dataset_base_path, 'mafood_labels.npy'), allow_pickle=True).item()

        # Cargar etiquetas single-label (platos)
        if self.data_split == 'train':
            lbls_d_file = os.path.join(dataset_base_path, 'train_lbls_d.txt')
            txt_file = os.path.join(dataset_base_path, 'train.txt')
        elif self.data_split == 'test':
            lbls_d_file = os.path.join(dataset_base_path, 'test_lbls_d.txt')
            txt_file = os.path.join(dataset_base_path, 'test.txt')
        else:
            raise ValueError(f"Unsupported data_split: {self.data_split}")

        # Cargar la lista de imágenes
        with open(txt_file, 'r') as f:
            self.image_list = [line.strip() for line in f]

        # Cargar la lista de labels de platos
        with open(lbls_d_file, 'r') as f:
            self.dish_labels = [int(line.strip()) for line in f]

        # Cargar nombres de los platos (para debugging o métricas por clase)
        dishes_file = os.path.join(dataset_base_path, 'dishes.txt')
        with open(dishes_file, 'r') as f:
            self.dish_names = [line.strip() for line in f]

        # Configuración de rutas de anotaciones
        if annFile == "":
            self.annFile = os.path.join(self.root, 'Annotations')
        else:
            raise NotImplementedError

        # Submuestreo para entrenamiento parcial (p porcentaje de datos)
        if self.data_split == 'train':
            num_examples = len(self.image_list)
            pick_example = int(num_examples * p)
            self.image_list = self.image_list[:pick_example]
            self.dish_labels = self.dish_labels[:pick_example]
        else:
            self.image_list = self.image_list
            self.dish_labels = self.dish_labels

        # Definición de las transformaciones para entrenamiento y prueba
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # ColorJitter suave, p.ej. ±0.2 en brillo y contraste
            RandomHorizontalFlip(p=0.5), # Flip horizontal con prob. 0.5
            CutoutPIL(cutout_factor=0.1), # Cutout menos agresivo (factor=0.3 en lugar de 0.5)
#            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # Selección de la transformación según el split
        if self.data_split == 'train':
            self.transform = train_transform
        elif self.data_split == 'test':
            self.transform = test_transform
        else:
            raise ValueError(f"data split = {self.data_split} is not supported in mafood")

        # Creación de la máscara para partial labels (si aplica)
        self.mask = None
        self.partial = partial
        if self.data_split == 'trainval' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.image_list), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask], dim=1)
                torch.save(mask, os.path.join(self.root, 'Annotations', f'partial_label_{partial:.2f}.pt'))
            else:
                mask = torch.load(os.path.join(self.root, 'Annotations', label_mask))
            self.mask = mask.long()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # Ruta de la imagen
        img_path = os.path.join('/media/HDD_4TB_2/daniel/gnn_test/3.MLR-GCN/MLR-GCN_MTL_6/food_datasets/MAFood121/images/', self.image_list[index])
        img = Image.open(img_path).convert('RGB')

        # Ingredientes (multi-label)
        label_vector = torch.tensor(self.labels_lab[self.image_list[index][:-4]])
        targets = label_vector.long()

        # Platos (single-label)
        dish_label = self.dish_labels[index]
        dish_target = torch.tensor(dish_label).long()

        # Aplicar máscara si corresponde
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            targets = self.mask[index] * targets + (1 - self.mask[index]) * masked

        # Aplicar transformaciones a la imagen
        if self.transform is not None:
            img = self.transform(img)

        # Devolver imagen, ingredientes (multi-label), plato (single-label)
        return img, targets[None, ], dish_target

    def name(self):
        return 'mafood'