import hashlib

def sha256(data):
    hash_obj = hashlib.sha256()
    hash_obj.update(data.encode())
    return hash_obj.hexdigest()


transactions = ['A', 'B', 'C', 'D']
tree = []

for t in transactions:
    tree.append(sha256(t))

# step 2 and 3: creating non-leaf nodes and forming the tree
while len(tree) > 1:
    temp = []
    for i in range(0, len(tree), 2):
        data = tree[i]
        if i + 1 < len(tree):
            data += tree[i+1]
        temp.append(sha256(data))
    tree = temp


# the remaining hash is the merkle root
merkle_root = tree[0]

print("merkle root: ", merkle_root)