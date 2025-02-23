import csv
import torch

# d = {}
# mode = 'eval'
# with open("./molecule-datasets/merge_final_" + mode + ".csv", 'r', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         # molecule,fragments,linker,product,class,center1,center2,anchors,linksize
#         mol_smiles = row['molecule']
#         prod_smiles = row['product']
#         linker_smiles = row['linker']
#         fragments_smiles = row['fragments']
#         c1 = row['center1']
#         c2 = row['center2']
#         anc = row['anchors']
#         linksize = row['linksize']

#         class_str = int(row['class'])

#         key = mol_smiles + '&' + fragments_smiles + '&' + prod_smiles + '&' + str(c1) + '&' + str(c2) +  '&' + str(anc) + '&' + str(linksize)
#         if key not in d:
#             d[key] = {}
            
#         d[key] = class_str

# # 66263 8345 8322
# torch.save(d, 'mapping_knownclass_' + mode + '.pt')
# d = torch.load('./mapping_knownclass_' + mode + '.pt')
# print(len(d.keys()))


# Merge
d1 = torch.load('./mapping_knownclass_' + 'eval' + '.pt')
d2 = torch.load('./mapping_knownclass_' + 'test' + '.pt')
d3 = torch.load('./mapping_knownclass_' + 'train' + '.pt')

def merge_dicts(*dicts):
    merged_dict = {}
    for d in dicts:
        for k, v in d.items():
            if k in merged_dict:
                if merged_dict[k] != v:
                    print(f"Warning: Key '{k}' has different values '{merged_dict[k]}' and '{v}'")
            else:
                merged_dict[k] = v
    return merged_dict

mapping_knownclass = merge_dicts(d1, d2, d3)
torch.save(mapping_knownclass, 'mapping_knownclass' + '.pt')

d = torch.load('./mapping_knownclass' + '.pt')
print(len(d.keys()))