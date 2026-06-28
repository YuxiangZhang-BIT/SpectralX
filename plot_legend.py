import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

WHUOHS_colors_list = [
    (0, 0, 0),          # Background
    (0, 0, 255),        # Paddy field
    (0, 255, 255),      # Dry farm
    (128, 128, 0),      # Woodland
    (128, 0, 0),        # Shrubbery
    (0, 0, 154),        # Sparse woodland
    (0, 255, 0),        # Other forest land
    (255, 0, 0),        # High-covered grassland
    (25, 25, 112),        # Medium-covered grassland
    (127, 255, 127),    # Low-covered grassland
    (216, 191, 216),    # River/canal
    (75, 0, 130),      # Lake
    (70, 130, 180),        # Reservoir/pond
    (240, 255, 255),    # Beach land
    (64, 224, 208),     # Shoal
    (34, 139, 34),        # Urban built-up
    (240, 230, 140),    # Rural settlement
    (245, 222, 179),      # Other construction land
    (255, 248, 220),    # Sand
    (255, 218, 185),    # Gobi
    (160, 82, 45),      # Saline/alkali soil
    (250, 128, 114),    # Marshland
    (255, 182, 203),    # Bare land
    (178, 34, 34),        # Bare rock
    (192, 192, 192)     # Ocean
]
WHUOHS_class_names = [
    "Background",
    "Paddy field",
    "Dry farm",
    "Woodland",
    "Shrubbery",
    "Sparse woodland",
    "Other forest land",
    "High-covered grassland",
    "Medium-covered grassland",
    "Low-covered grassland",
    "River/canal",
    "Lake",
    "Reservoir/pond",
    "Beach land",
    "Shoal",
    "Urban built-up",
    "Rural settlement",
    "Other construction land",
    "Sand",
    "Gobi",
    "Saline/alkali soil",
    "Marshland",
    "Bare land",
    "Bare rock",
    "Ocean"
]

# DFC_colors_list = [
#     (0, 0, 0),       
#     (255, 0, 0),    
#     (0, 255, 0), 
#     (0, 0, 255),      
#     (255, 255, 0),  
#     (255, 0, 255),  
#     (0, 255, 255), 
#     (128, 0, 0),   
#     (0, 128, 0)  
# ]
# DFC_class_names = [ 
#     "Background", 
#     'Forest', 
#     'Shrubland', 
#     'Grassland', 
#     'Wetlands', 
#     'Croplands', 
#     'Urban/Build-up', 
#     'Barren' , 
#     'Water'
# ]

# MTS12_colors_list = [
#     (0, 0, 0),       
#     (255, 0, 0),    
#     (0, 255, 0), 
#     (0, 0, 255),      
#     (255, 255, 0),  
#     (255, 0, 255),  
#     (0, 255, 255), 
#     (128, 0, 0),   
#     (0, 128, 0)  
# ]
# MTS12_class_names = [
#     'Cultivated land',
#     'Forest',
#     'Grassland',
#     'Shrubland',
#     'Water',
#     'Wetland',
#     'Artificial surface',
#     'Bare land'
# ]


colors_list = WHUOHS_colors_list
cmap_all = mpl_colors.ListedColormap(np.array(colors_list) / 255.0)

fig, ax = plt.subplots(figsize=(6, 12))
for i, color in enumerate(colors_list):
    ax.add_patch(plt.Rectangle((0, i), 1, 1, facecolor=np.array(color) / 255.0, edgecolor='none'))
    
    ax.text(1.5, i + 0.5, WHUOHS_class_names[i], ha='left', va='center', fontsize=10)

ax.set_xlim(0, 3)
ax.set_ylim(len(colors_list), 0)
ax.axis('off')
plt.show()
plt.savefig(f'WHUOHS_legend_1.png', dpi=100, bbox_inches='tight')
