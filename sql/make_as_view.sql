BEGIN TRANSACTION;

-- 1. Create the new table with the Primary Key definition
-- Make sure the data types (INTEGER/TEXT) match your original schema
CREATE TABLE text_rank_thread_new (
    mid INTEGER PRIMARY KEY,
    text_rank_query TEXT
    elaborative_description TEXT
);

-- 2. Copy the data for the columns you want to keep
INSERT INTO text_rank_thread_new (mid, text_rank_query, elaborative_description)
SELECT mid, text_rank_query, elaborative_description
FROM text_rank_thread;

-- 3. Drop the old table
DROP TABLE text_rank_thread;

-- 4. Rename the new table to the original name
ALTER TABLE text_rank_thread_new RENAME TO text_rank_thread;

COMMIT;

-- 5. Reclaim disk space (optional but recommended)
VACUUM;
