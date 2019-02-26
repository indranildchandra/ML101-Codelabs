import collections

stepSize = 0.01 # learning rate


def read_data() :
    data = open("./../resources/vehicle_sale_data.csv" , "r")
    gdp_sale = collections.OrderedDict()
    for line in data.readlines()[1:] :
        record = line.split(",")
        gdp_sale[float(record[1])] = float(record[2].replace('\n', ""))
    print(gdp_sale)
    return gdp_sale


def sale_for_data(constant, slope, data):
    return constant + slope * data   # y = c + ax format


def step_cost_function_for(gdp_sale, constant, slope) :
    global stepSize
    diff_sum_constant = 0 # diff of sum for constant 'c' in "c + ax" equation
    diff_sum_slope = 0  # diff of sum for 'a' in "c + ax" equation
    gdp_for_years = list(gdp_sale.keys())

    for year_gdp in gdp_for_years: # for each year's gdp in the sample data
        # get the sale for given 'c' and 'a'by giving the GDP for this sample record
        trg_data_sale = sale_for_data(constant, slope, year_gdp) # calculated sale for current 'c' and 'a'
        a_year_sale = gdp_sale.get(year_gdp) # real sale for this record
        diff_sum_slope = diff_sum_slope + ((trg_data_sale - a_year_sale) * year_gdp) # slope is (h(y) - y) * x
        diff_sum_constant = diff_sum_constant + (trg_data_sale - a_year_sale) # constant is (h(y) - y)

    step_for_constant = (stepSize / len(gdp_sale)) * diff_sum_constant # distance to be moved by c
    step_for_slope = (stepSize / len(gdp_sale)) * diff_sum_slope # distance to be moved by a
    new_constant = constant - step_for_constant # new c
    new_slope = slope - step_for_slope # new a

    return new_constant, new_slope


def get_weights(gdp_sale) :
    constant = 1
    slope = 1
    accepted_diff = 0.01

    while 1 == 1:  # continue till we reach local minimum
        new_constant, new_slope = step_cost_function_for(gdp_sale, constant, slope)
        # if the diff is too less then lets break
        if (abs(constant - new_constant) <= accepted_diff) and (abs(slope - new_slope) <= accepted_diff):
            print("Difference between values in last iteration and current iteration for both constant and slope are less than " + str(accepted_diff))
            return new_constant, new_slope
        else:
            constant = new_constant
            slope = new_slope
            print("Updated values: Constant = " + str(new_constant) + ", Slope = " + str(new_slope))


def main():
    contant, slope = get_weights(read_data())
    print("Final values: Constant : " + str(contant) + ", Slope:" + str(slope))

if __name__ == '__main__':
    main()
