import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

# ── CONFIGURACIÓN ─────────────────────────────────────
MODELO_PATH = 'best_model_multimodal.pth'
IMSIZE      = 64
device      = torch.device('cpu')

# Estadísticas de normalización del dataset de entrenamiento
# (valores calculados durante el entrenamiento)
FEAT_MEAN = np.array([3.74, 5.67, 3.80, 3.90, 1.60, 1.53, 4.35, 2.74], dtype=np.float32)
FEAT_STD  = np.array([1.18, 0.93, 0.94, 1.24, 0.97, 0.95, 1.23, 1.08], dtype=np.float32)

# ── MODELO MULTIMODAL ─────────────────────────────────
class MultimodalNet(nn.Module):
    def __init__(self, n_features=8):
        super().__init__()
        backbone = models.resnet18(weights=None)
        for name, param in backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        backbone.fc = nn.Identity()
        self.imagen_encoder   = backbone
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 32),       nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid()
        )

    def forward(self, img, feat):
        return self.fusion(torch.cat([
            self.imagen_encoder(img),
            self.clinical_encoder(feat)
        ], dim=1))

def cargar_modelo():
    model = MultimodalNet(n_features=8)
    model.load_state_dict(torch.load(MODELO_PATH, map_location=device))
    model.eval()
    return model

# ── HELPERS ───────────────────────────────────────────
def nivel_riesgo(score):
    if score < 0.33:   return "BAJO",  "#2ecc71", "Control rutinario"
    elif score < 0.66: return "MEDIO", "#f39c12", "Seguimiento recomendado"
    else:              return "ALTO",  "#e74c3c", "Evaluacion urgente"

def preprocesar_imagen(img_path):
    img = Image.open(img_path).convert('L').resize((IMSIZE, IMSIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(np.stack([img, img, img], axis=0)).unsqueeze(0)

def preprocesar_features(features_dict):
    feat = np.array([
        features_dict['subtlety'],
        features_dict['calcification'],
        features_dict['sphericity'],
        features_dict['margin'],
        features_dict['lobulation'],
        features_dict['spiculation'],
        features_dict['texture'],
        features_dict['malignancy'],
    ], dtype=np.float32)
    feat = (feat - FEAT_MEAN) / FEAT_STD
    return torch.tensor(feat).unsqueeze(0)

# ── FORMULARIO INTERACTIVO ────────────────────────────
def pedir_datos_nodulo():
    print("\n" + "="*55)
    print("  ONCASCAN PLATFORM — Evaluacion de Nodulo Pulmonar")
    print("="*55)
    print("\nIngrese los parametros radiologicos del nodulo (1-5):")
    print("(Escala: 1=minimo  5=maximo)\n")

    campos = [
        ("subtlety",      "Sutileza        (dificultad para ver el nodulo)", 1, 5),
        ("calcification", "Calcificacion   (presencia de calcio)",           1, 6),
        ("sphericity",    "Esfericidad     (que tan redondo es)",             1, 5),
        ("margin",        "Margen          (que tan definido es el borde)",   1, 5),
        ("lobulation",    "Lobulacion      (forma lobulada)",                 1, 5),
        ("spiculation",   "Espiculacion    (picos o espiculas en el borde)",  1, 5),
        ("texture",       "Textura         (1=vidrio esmerilado 5=solido)",   1, 5),
        ("malignancy",    "Malignidad      (sospecha visual del radiologo)",  1, 5),
    ]

    features = {}
    for key, descripcion, vmin, vmax in campos:
        while True:
            try:
                val = float(input(f"  {descripcion} [{vmin}-{vmax}]: "))
                if vmin <= val <= vmax:
                    features[key] = val
                    break
                else:
                    print(f"  Valor fuera de rango. Ingrese entre {vmin} y {vmax}.")
            except ValueError:
                print("  Ingrese un numero valido.")

    return features

def seleccionar_imagen():
    print("\nImagenes CT disponibles en la carpeta:")
    imagenes = [f for f in os.listdir('.') if f.endswith('.png')
                and not f.startswith('resultado')]
    for i, img in enumerate(imagenes):
        print(f"  [{i+1}] {img}")

    while True:
        try:
            idx = int(input("\nSeleccione imagen [numero]: ")) - 1
            if 0 <= idx < len(imagenes):
                return imagenes[idx]
            print("Numero fuera de rango.")
        except ValueError:
            print("Ingrese un numero valido.")

# ── VISUALIZACIÓN RESULTADO ───────────────────────────
def mostrar_resultado(img_path, features, score, model):
    nivel, color, recomendacion = nivel_riesgo(score)
    img_orig = np.array(Image.open(img_path).convert('L'))

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#0A1628')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Título
    fig.suptitle('OncaScan Platform — Resultado del Analisis',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    # Panel 1: imagen CT
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img_orig, cmap='gray')
    ax1.set_title('CT Pulmonar', color='white', fontsize=12, pad=10)
    ax1.axis('off')
    ax1.set_facecolor('#0A1628')

    # Panel 2: resultado principal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#112240')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.05, 0.55), 0.9, 0.38,
                  color=color, alpha=0.15, linewidth=2, edgecolor=color))
    ax2.text(0.5, 0.85, f'RIESGO {nivel}',
             ha='center', va='center', fontsize=20,
             fontweight='bold', color=color)
    ax2.text(0.5, 0.67, f'Score: {score:.4f}',
             ha='center', va='center', fontsize=14, color=color)
    ax2.text(0.5, 0.38, recomendacion,
             ha='center', va='center', fontsize=11,
             color='white', style='italic')
    ax2.text(0.5, 0.18, f'Archivo: {os.path.basename(img_path)}',
             ha='center', va='center', fontsize=9, color='gray')
    ax2.set_title('Diagnostico IA', color='white', fontsize=12, pad=10)

    # Panel 3: medidor de riesgo
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#112240')
    for ancho, c, left in zip(
        [0.33, 0.33, 0.34],
        ['#2ecc71', '#f39c12', '#e74c3c'],
        [0, 0.33, 0.66]
    ):
        ax3.barh(0, ancho, left=left, color=c, height=0.4, alpha=0.8)
    ax3.barh(0, 0.015, left=max(0, score-0.007),
             color='white', height=0.7)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.6, 0.9)
    ax3.set_yticks([])
    ax3.set_xlabel('Score de riesgo', color='white')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('gray')
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_facecolor('#112240')
    ax3.text(0.165, 0.38, 'BAJO',  ha='center', fontsize=9,
             fontweight='bold', color='#27ae60')
    ax3.text(0.495, 0.38, 'MEDIO', ha='center', fontsize=9,
             fontweight='bold', color='#e67e22')
    ax3.text(0.825, 0.38, 'ALTO',  ha='center', fontsize=9,
             fontweight='bold', color='#c0392b')
    ax3.text(score, -0.45, f'{score:.2f}',
             ha='center', fontsize=10, fontweight='bold', color='white')
    ax3.set_title('Nivel de riesgo', color='white', fontsize=12, pad=10)

    # Panel 4: features clínicas ingresadas
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor('#112240')
    ax4.axis('off')

    nombres = ['Sutileza', 'Calcif.', 'Esferici.', 'Margen',
               'Lobulac.', 'Espicul.', 'Textura', 'Malignid.']
    valores  = [features['subtlety'],      features['calcification'],
                features['sphericity'],    features['margin'],
                features['lobulation'],    features['spiculation'],
                features['texture'],       features['malignancy']]
    maximos  = [5, 6, 5, 5, 5, 5, 5, 5]

    x     = np.arange(len(nombres))
    barras = ax4.barh(x, valores,
                      color=[color if v/m >= 0.6 else '#378ADD'
                             for v, m in zip(valores, maximos)],
                      alpha=0.8, height=0.6)
    for i, (v, m) in enumerate(zip(valores, maximos)):
        ax4.barh(i, m, color='gray', alpha=0.2, height=0.6)
        ax4.text(v + 0.05, i, f'{v:.1f}/{m}',
                 va='center', fontsize=9, color='white')

    ax4.set_yticks(x)
    ax4.set_yticklabels(nombres, color='white', fontsize=10)
    ax4.set_xlim(0, 7)
    ax4.set_xlabel('Valor ingresado por el radiologo', color='white')
    ax4.tick_params(colors='white')
    ax4.spines['bottom'].set_color('gray')
    ax4.spines['top'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_facecolor('#112240')
    ax4.set_title('Parametros radiologicos ingresados', color='white',
                  fontsize=12, pad=10)

    plt.savefig(f'resultado_endtoend_{os.path.basename(img_path)}',
                dpi=150, bbox_inches='tight', facecolor='#0A1628')
    plt.show()

# ── MAIN ──────────────────────────────────────────────
if __name__ == '__main__':
    print("\nCargando OncaScan Platform...")
    model = cargar_modelo()
    print("✓ Modelo multimodal listo\n")

    while True:
        # Seleccionar imagen
        img_path = seleccionar_imagen()

        # Ingresar datos clínicos
        features = pedir_datos_nodulo()

        # Inferencia
        img_tensor  = preprocesar_imagen(img_path)
        feat_tensor = preprocesar_features(features)

        with torch.no_grad():
            score = model(img_tensor, feat_tensor).item()

        nivel, color, rec = nivel_riesgo(score)

        print(f"\n{'='*55}")
        print(f"  RESULTADO ONCASCAN")
        print(f"{'='*55}")
        print(f"  Score:          {score:.4f}")
        print(f"  Nivel de riesgo: {nivel}")
        print(f"  Recomendacion:  {rec}")
        print(f"{'='*55}")

        mostrar_resultado(img_path, features, score, model)

        otra = input("\n¿Analizar otro caso? (s/n): ").strip().lower()
        if otra != 's':
            print("\nOncaScan Platform — Sesion terminada.")
            break