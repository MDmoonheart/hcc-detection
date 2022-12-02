from re import L
import mysql.connector
from numpy import imag

class DBconnector:

    conn = None
    cursor = None

    '''
    Construor of DBconnector:
    Input: 
    user: str. The username of database user.
    host: str. The hostname of database server. Default: 127.0.0.1
    port: str. The port of database server. Default: 3306
    password: str. The password of database user. 
    database: str. The name of database used.

    Output: None
    After using the constructor, it established the connection between the py code and database.
    '''
    def __init__(self, user, host, port, password, database):
        try:
            conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
            )
            
            self.conn = conn
            self.cursor = conn.cursor

        except AttributeError:
            print(AttributeError)
    
    '''
    execute_sql_query_command(self, sql)
    Used to execuate query command of sql. Auto commited.
    Input: 
    sql: The sql command.

    Output:
    The query results in list of tuples.

    '''
    def execute_sql_query_command(self, sql):
        print("Query: ", sql)
        
        my_cursor = self.get_cursor()
 
        my_cursor.execute(sql)

        my_result = my_cursor.fetchall()     # fetchall() 获取所有记录
        
        return my_result

    '''
    execuate_sql_update_command(self, sql):
    Used to update the results with command. Auto commited.
    Execuate sql query command
    sql: string. The input 
    '''
    def execuate_sql_update_command(self, sql):
        print("SQL: ", sql)
        self.cursor().execute(sql)
        self.conn.commit()

    ''' Get the cursor of the connection.

    Returns:
    The cursor of the connection.
    '''
    def get_cursor(self):
        return self.conn.cursor(buffered=True)
    
    ''' Create a new dataset 
    '''
    def insert_dataset(self, is_public: int, path: str, src_name: str, src_desc: str, sid: int=0):
        path = path.replace('\\', '/')
        if sid == 0:
            sql = "insert into src_info (is_public, path, src_name, src_desc, src_size) values (%d, '%s', '%s', '%s', %d)" % (is_public, path, src_name, src_desc, 0)
        else:
            sql = "insert into src_info (sid, is_public, path, src_name, src_desc, src_size) values (%d, %d, '%s', '%s', '%s', %d)" % (sid, is_public, path, src_name, src_desc, 0)
            
        self.execuate_sql_update_command(sql)

    def insert_case_record(self, cid, case_name, case_ref, source, import_date, extra_info, case_type, gender, relative_path='/'):
        sql = "insert into cases values (%d, '%s', '%s', %d, '%s', '%s', %d, %d, '%s')" % (cid, case_name, case_ref, source, import_date, extra_info, case_type, gender, relative_path)
        print(sql)
        self.execuate_sql_update_command(sql)

    def get_dataset_id(self, src_name: str):
        try:
            sql = "select sid from src_info where src_name = '%s' " % (src_name)
            print(sql)
            res = self.execute_sql_query_command(sql)
            if(len(res) == 0):
                return -1
            return res[0][0]
        
        except mysql.connector.Error as err:
            print("Error: ", type(err))
            return -1

    def case_exists(self, case_name: str):
        sql = "select count(*) from cases where case_name = '%s' " % (case_name)
        res = self.execute_sql_query_command(sql)

        return res[0][0] > 0

    def get_last_dataset_id(self):
        sql = "select max(sid) from src_info"
        res = self.execute_sql_query_command(sql)

        if(res[0][0] == None):
            return 0

        return res[0][0]
    
    def get_case_id(self, case_name: str):
        try:
            sql = "select cid from cases where case_name = '%s' " % (case_name)
            print(sql)
            res = self.execute_sql_query_command(sql)
            if(len(res) == 0):
                return -1
            return res[0][0]
        
        except mysql.connector.Error as err:
            print("Error: ", type(err))
            return -1
    
    

    def get_absolute_path_with_case_name(self, case_name: str):
        pass

    def get_all_cases_with_dataset_id(self, sid: int):
        sql = " select cid, case_name from cases inner join src_info on cases.source = src_info.sid where src_info.sid = %d" % (sid)
        res = self.execute_sql_query_command(sql)

        return list(res)

    def get_dataset_absolute_path_with_sid(self, sid: int):
        sql = "select path from src_info where sid = %d" % (sid)
        res = self.execute_sql_query_command(sql)

        print(res)

        return list(res)

    def get_dataset_basic_info_with_sid(self, sid: int):
        sql = "select is_public, src_desc from src_info where sid = %d" % (sid)
        res = self.execute_sql_query_command(sql)

        return list(res)

    def get_case_basic_info_with_cid(self, cid:id):
        sql = "select case_ref, source, import_date, extra_info, case_type, gender from cases where cid = %d" % (cid)
        res = self.execute_sql_query_command(sql)

        return list(res) 

    def get_all_case_info_with_case_id(self, case_id: int):
        sql = "select case_meta_id, case_id, sub_path from case_info where case_id = %d" % (case_id)
        res = self.execute_sql_query_command(sql)

        return list(res)

    def get_all_cases_path_with_zero_thickness(self):
        sql = " Select case_meta_id, concat(path,  relative_path, '/', case_name, sub_path) \
                from case_info inner join cases \
                on case_info.case_id=cases.cid \
                inner join src_info \
                on cases.source=src_info.sid \
                where thickness=0 and case_info.img_count>1;"

        return self.execute_sql_query_command(sql)
    '''
    insert_ct_info: Used to insert a ct_series with the specfic case.
    meta_id: uuid. 
    '''
    
    
    def insert_ct_info(self, meta_id, case_id, is_contrasted, phase, additional_info, image_count, width=512, height=512, thickness=1.25, subpath='/', has_tumor=0, lifetime=0, tumor_info=[0,0,0]):
        sql = "insert into case_info values ('%s', %d, %d, %d, '%s', %d, %d, %d, %f, '%s', %d, %d, %f, %f, %f) " \
                % \
                (meta_id, case_id, is_contrasted, phase, additional_info, image_count, width, height, thickness, subpath, has_tumor, lifetime, tumor_info[0], tumor_info[1], tumor_info[2])
        
        print("Insert CT info to the database...")
        print(sql)
        self.execuate_sql_update_command(sql)

    def delete_dataset_with_sid(self, sid: int):
        sql = "delete from src_info where sid = %d" % (sid)
        self.execuate_sql_update_command(sql)

    def delete_case_with_cid(self, cid: int):
        sql = "delete from cases where cid = %d" % (cid)
        self.execuate_sql_update_command(sql)

    def delete_case_info_with_case_meta_id(self, case_meta_id: str):
        sql = "delete from case_info where case_meta_id = '%s' " % (case_meta_id)
        self.execuate_sql_update_command(sql)

    def get_absolute_path_of_cases(self, cid: int):
        sql = "Select concat(path,  relative_path) as image_path \
               From cases \
               inner join src_info \
               on source = sid \
               Where cid = '%d' " % (cid)
        
        res = self.execute_sql_query_command(sql)
        return res
        

    def get_absolute_path_of_all_case_info(self, cid: int):
        sql = "Select concat(path,  relative_path,  sub_path) as image_path \
               From case_info \
	           inner join cases \
 	           on case_id = cid \
               inner join src_info \
               on source = sid \
               Where cid = '%d' " % (cid)
        
        res = self.execute_sql_query_command(sql)

        return res

    def disconnect(self):
        self.conn.disconnect()
        