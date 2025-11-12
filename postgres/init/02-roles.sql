-- Создаём пользователя приложения
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'embedd_user') THEN
    CREATE USER embedd_user WITH PASSWORD 'embedd_password';
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'dev') THEN
    CREATE USER dev WITH PASSWORD 'dev_password';
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rerank_user') THEN
    CREATE USER rerank_user WITH PASSWORD 'rerank_password';
  END IF;
END
$$;

ALTER USER embedd_user WITH SUPERUSER; -- изменить на нужные права потом
ALTER USER dev WITH SUPERUSER;
ALTER USER rerank_user WITH SUPERUSER;