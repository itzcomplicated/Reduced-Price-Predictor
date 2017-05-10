from sklearn import tree


def float_to_string(float):
    return "%0.2f" % float


def int_to_string(integer):
    return "%d" % integer


classifier = tree.DecisionTreeClassifier()

# Keeping a sample successful input&output set. This should be of more size and
# should be frequently updated for better results.
# training_input format :: current price * 100 (for making it integer), days to expire *10, quantity
training_input = [[550, 20, 5],
                  [450, 10, 20],
                  [200, 10, 15],
                  [100, 20, 3],
                  [150, 30, 25],
                  [220, 20, 8],
                  [250, 10, 5],
                  [500, 10, 20],
                  [300, 10, 12],
                  [225, 20, 9],
                  [350, 30, 14],
                  [125, 10, 2],
                  [150, 10, 16],
                  [200, 20, 6],
                  [375, 10, 10],
                  [675, 20, 15],
                  [800, 20, 5],
                  [450, 20, 30],
                  [450, 10, 30],
                  [450, 10, 15],
                  [400, 20, 1]]
# training_output format :: recommended price * 100 (for making it integer)
training_output = [499, 211, 199, 99, 129, 199, 219, 299, 199, 190, 300, 115, 79, 190, 249, 499, 699, 269, 199, 239, 390]

# Training
classifier = classifier.fit(training_input, training_output)

print("\nReduced price recommendation system...\n")

current_price = input("Current price : ")
current_price = float(current_price)
current_price = int(current_price * 100)

days_to_expire = input("Number of days to expire : ")
days_to_expire = int(days_to_expire)
days_to_expire = days_to_expire*10

items_left = input("Number of items left : ")
items_left = int(items_left)

recommended_price_np = classifier.predict([[current_price, days_to_expire, items_left]])

print("=========================================================================================")

current_price = current_price / 100
days_to_expire = days_to_expire / 10
print("\nCurrent price : " + float_to_string(current_price))
print("Days to expire : " + int_to_string(days_to_expire))
print("Items left : " + int_to_string(items_left))

recommended_price = recommended_price_np[0] / 100
recommended_price=float(recommended_price)

if recommended_price > current_price:
    print("Prediction not available!")
else:
    print("*** Recommended Price is : " + float_to_string(recommended_price))

print("=========================================================================================")
