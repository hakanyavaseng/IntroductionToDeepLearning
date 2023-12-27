import matplotlib.pyplot as plt

     

#Function to read
def readFile(path):
  with open(path, "r", encoding="utf-8") as f:
    txtFile = f.read().lower()
    return txtFile

#Reads spor, ekonomi and siyaset text files.
txtSpor = readFile("/content/spor.txt")
txtEkonomi = readFile("/content/ekonomi.txt")
txtSiyaset = readFile("/content/siyaset.txt")

# Read secilenler text file
with open('/content/secilenler.txt') as f:
    secilenler = f.read().splitlines()
#print(secilenler)


# Her belgedeki kelimelerin sayısını hesapla
spor = [txtSpor.count(word) for word in secilenler]
ekonomi = [txtEkonomi.count(word) for word in secilenler]
siyaset = [txtSiyaset.count(word) for word in secilenler]


# Her belgedeki kelimelerin sayısını gösteren bir çubuk grafiği oluştur
wordIndexes = range(len(secilenler))

# Spor
plt.figure(figsize=(10, 2))
plt.bar(wordIndexes, spor, color='y', align='center')
plt.xlabel('Kelimeler')
plt.ylabel('Tekrarlar')
plt.title('Spor')
plt.xticks(wordIndexes, secilenler, rotation='vertical')
plt.show()

# Ekonomi
plt.figure(figsize=(10, 2))
plt.bar(wordIndexes, ekonomi, color='r', align='center')
plt.xlabel('Kelimeler')
plt.ylabel('Tekrarlar')
plt.title('Ekonomi')
plt.xticks(wordIndexes, secilenler, rotation='vertical')
plt.show()

# Siyaset
plt.figure(figsize=(10, 2))
plt.bar(wordIndexes, siyaset, color='g', align='center')
plt.xlabel('Kelimeler')
plt.ylabel('Tekrarlar')
plt.title('Siyaset')
plt.xticks(wordIndexes, secilenler, rotation='vertical')
plt.show()
     