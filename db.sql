CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    c_name TEXT NOT NULL,
    c_surname TEXT NOT NULL,
    license_number TEXT NOT NULL,
    jop_position TEXT NOT NULL
);

#15
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Alina","Zhamalova","188ADT17","Financial analyst");
#13
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Maksat","Kanatov","898ABS02","Manager");

#46
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Kanat","Sakenovich","797FFA02","Executive Assistant");

#70
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Rakhat","Imanzholov","969VCB05","Financial analyst");

#42
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Arman","Amanzholovich","025UUU05","MArketing analyst");

#56
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Shynar","Kainarova","434JJA02","HR");

#19
INSERT INTO CUSTOMERS(c_name,c_surname,license_number,jop_position)
VALUES("Zhazira","Kainarova","860ACK02","HR");



Select * from customers;

Create database data;

drop table customers;