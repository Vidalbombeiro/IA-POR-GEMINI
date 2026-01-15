#from sklearn import tree

# 1. DADOS (O Combustível)
# Aqui definimos as características: [Peso, Textura]
# Vamos convencionar: 0 = Lisa, 1 = Irregular
features = [[140, 0], [130, 0], [150, 1], [170, 1]]

# Aqui definimos os Rótulos (o que é cada fruta acima)
# Vamos convencionar: 0 = Maçã, 1 = Laranja
labels = [0, 0, 1, 1]

# 2. ESCOLHA DO ALGORITMO (O Motor)
# Vamos usar uma "Árvore de Decisão" (Decision Tree)
clf = tree.DecisionTreeClassifier()

# 3. TREINAMENTO (O Aprendizado)
# O método .fit() é onde a mágica acontece.
# Ele encontra padrões entre as 'features' e os 'labels'.
clf = clf.fit(features, labels)

# 4. PREVISÃO (O Teste)
# Vamos perguntar para a IA: "O que é uma fruta de 160g com textura irregular (1)?"
resultado = clf.predict([[160, 1]])

# Traduzindo o resultado para nós
if resultado == 0:
    print("A IA acha que é uma Maçã!")
else:
    print("A IA acha que é uma Laranja!")
