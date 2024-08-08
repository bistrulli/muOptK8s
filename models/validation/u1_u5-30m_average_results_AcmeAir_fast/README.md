## Results
The files 'exp_name_avg_matrix.csv' are formatted as follows.

### Rows
Each row corresponds to a microservice:

- 0 - Booking BookFlights
- 1 - Booking CancelBooking
- 2 - Customer ByIdGET
- 3 - Customer ByIdPOST
- 4 - Customer UpdateMiles
- 5 - Customer ValidateId
- 6 - Flight QueryFlights
- 7 - Flight GetReWardMiles
- 8 - Auth

### Columns
Each column corresponds to a metric:

- 0 - CPU_Usage (GoogleCloud metric)
- 1 - Replicas (GoogleCloud metric)
- 2 - RPS (Custom metric)
- 3 - Response_Time (Custom metric)
- 3 - Service_Time (Estimated with Little's Law)