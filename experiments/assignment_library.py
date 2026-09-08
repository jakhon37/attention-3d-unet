# Python 3

number_list = []
n = int(input("Enter the list size "))

print("\n")
for i in range(0, n):
    print("Enter number at index", i, )
    item = int(input())
    number_list.append(item)
print("User list is ", number_list)







txt = input("Type something to test this out: ")

# Note that in version 3, the print() function
# requires the use of parenthesis.
print("Is this what you just said? ", txt)

# An input is requested and stored in a variable
test_text = input ("Enter a number: ")

# Converts the string into a integer. If you need
# to convert the user input into decimal format,
# the float() function is used instead of int()
test_number = int(test_text)

# Prints in the console the variable as requested
print ("The number you entered is: ", test_number)