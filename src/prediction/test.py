from sqlalchemy import create_engine
import psycopg2
import tensorflow as tf
# import tensorflow as tf 

if __name__ == "__main__":
    db_user = 'postgres'
    db_password = 'oX7IDNsZF1OrTOzS75Ek'
    db_host = 'database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com'
    db_port = '5432'
    db_name = 'capstone_project'
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    print("engine creted")
    try:
        with engine.connect() as conn:
            print("connected")
    except Exception:
        print("failed")