# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:43:09 2018

@author: chengsijin
"""

import pymysql.cursors


def connect():
    
    """
    Creates a new database session and belonging cursor for command execution.
    Executes a statement and prints database server version before attempting to close connection
    """
    connection = None
    try:
        print('Connecting to the mySQL database...')
        # Connect to the database
        connection = pymysql.connect("localhost","root","","patentRecom")
        # create a cursor
        cursor = connection.cursor()      
        # execute a statement
        print('mySQL database version:')
        cursor.execute('SELECT version()')
        # display the mySQL database server version
        db_version = cursor.fetchone()
        print(db_version)
        # close the communication with the PostgreSQL
        cursor.close()
    except (Exception, pymysql.DatabaseError) as error:
        print(error)
    finally:
        if connection is not None:
            connection.close()
            print('Database connection closed.')

if __name__ == '__main__':
    connect()