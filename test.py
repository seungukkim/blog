class Study:
    def print_numbers(self, a,b,c ):
        print(a)
        print(b)
        print(c)

    def enumeratenums(self,*args):
        for arg in args:
            print(arg)


study1=Study()
study1.print_numbers(10,20,30)

x=[10,20,30]
study1.print_numbers(*x)

study1.enumeratenums(1,2,3,4,56,76,7)
