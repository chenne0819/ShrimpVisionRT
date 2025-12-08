import pymysql

# 连接数据库
connection = pymysql.connect(
    host='120.110.113.144',  # 数据库主机地址
    user='Shrimp',  # 数据库用户名
    password='chenne50515',  # 数据库密码
    database='Shrimp',  # 数据库名称
    
)

try:
    # 执行 SQL 查询
    with connection.cursor() as cursor:
        # 查询示例：获取所有记录
        sql = "SELECT * FROM ObjectVideos"
        cursor.execute(sql)
        result = cursor.fetchall()
        
        # 处理查询结果
        if result:
            for row in result:
                print(row)  # 打印每条记录的内容
        else:
            print("No records found.")

finally:
    # 关闭数据库连接
    connection.close()
