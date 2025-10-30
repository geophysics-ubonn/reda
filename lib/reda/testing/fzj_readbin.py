import reda
import pandas as pd
import io


def test_find_swapped_measurement_indices():

    json_obj = io.StringIO()
    json_obj.write(
        '{"index":{"0":0,"1":23},"delay":{"0":0.1,"1":0.1},"nr_samples":{"0":3000,"1":3000},"frequency":{"0":0.1,"1":0.1},"sampling_frequency":{"0":1000.0,"1":1000.0},"oversampling":{"0":10.0,"1":10.0},"U0":{"0":9.0,"1":9.0},"inj_number":{"0":1.0,"1":2.0},"a":{"0":1,"1":10},"b":{"0":2,"1":1},"timestamp":{"0":3700034644.7634205818,"1":3700034747.1302757263},"datetime":{"0":1617189844763,"1":1617189947130},"fa":{"0":100.0,"1":100.0},"tmax":{"0":30.0,"1":30.0}}'
    )
    json_obj.seek(0)

    # #########################################################################
    # 1. only one row in frequency data
    obj = reda.fzj_readbin()
    # two lines of data, but only one matches the (a, b, frequency) combination
    obj.frequency_data = pd.read_json(json_obj)
    indices = obj.find_swapped_measurement_indices(
        1, 2, 0.1, pd.Timestamp('2021-03-31 11:24:20.763421')
    )
    assert indices == [0, ]

    # #########################################################################
    # reverse order of frequency data

    obj.frequency_data = obj.frequency_data.sort_values(
        'datetime', ascending=False).reset_index()
    indices = obj.find_swapped_measurement_indices(
        1, 2, 0.1, pd.Timestamp('2021-03-31 11:24:20.763421')
    )
    assert indices == [1, ]

    # #########################################################################
    # again,only one relevant measurement
    obj = reda.fzj_readbin()

    json_obj2 = io.StringIO()
    json_obj2.write(
        '{"delay":{"0":0.1,"1":0.1,"2":0.1,"3":0.1,"4":0.1,"5":0.1},"nr_samples":{"0":3000,"1":3000,"2":3000,"3":3000,"4":3000,"5":3000},"frequency":{"0":0.1,"1":0.31446541,"2":0.89285714,"3":1.0,"4":1.0989011,"5":3.125},"sampling_frequency":{"0":1000.0,"1":1257.8616,"2":1785.7143,"3":1000.0,"4":1098.9011,"5":1041.6667},"oversampling":{"0":10.0,"1":4.0,"2":2.0,"3":1.0,"4":2.0,"5":1.0},"U0":{"0":9.0,"1":9.0,"2":9.0,"3":9.0,"4":9.0,"5":9.0},"inj_number":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0},"a":{"0":1,"1":1,"2":1,"3":1,"4":1,"5":1},"b":{"0":2,"1":2,"2":2,"3":2,"4":2,"5":2},"timestamp":{"0":3700034644.7634205818,"1":3700034655.9600605965,"2":3700034660.9313459396,"3":3700034665.6016125679,"4":3700034672.7090187073,"5":3700034677.2532792091},"datetime":{"0":1617189844763,"1":1617189855960,"2":1617189860931,"3":1617189865601,"4":1617189872709,"5":1617189877253},"fa":{"0":100.0,"1":314.4654,"2":892.85715,"3":1000.0,"4":549.45055,"5":1041.6667},"tmax":{"0":30.0,"1":9.5400002671,"2":3.3599999731,"3":3.0,"4":5.4599999945,"5":2.8799999078}}'
    )
    json_obj2.seek(0)
    obj.frequency_data = pd.read_json(json_obj2)
    indices = obj.find_swapped_measurement_indices(
        1, 2, 0.1, pd.Timestamp('2021-03-31 11:24:20.763421')
    )
    assert indices == [0, ]

test_find_swapped_measurement_indices()
