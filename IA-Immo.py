import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Création d'un échantillon de 200 valeurs
m = 200
x = np.array([740,880,650,2630,1895])
y = np.array([25,35,19,103,67])

# Création d'un modèle de régression
model = LinearRegression()
# Entraînement du modèle
model.fit(x.reshape(-1, 1), y)

print("Le score du modèle est de :", model.score(x.reshape(-1, 1), y))
print("Avec 1000€ vous pouvez prétendre à un appartement de", model.predict(np.array(1000).reshape(-1, 1)), "m²")

# Modélisation d'un graphique à l'aide des deux tableaux
plt.scatter(x, y)
plt.plot(x.reshape(-1, 1), model.predict(x.reshape(-1, 1)), c = 'red')
plt.title("Prix d'un loyer en fonction d'une surface")
plt.xlabel("Loyer mensuel en €")
plt.ylabel("Surface en m²")
plt.show()
