CREATE TABLE experiment_results (
    system TEXT,
    size TEXT,
    experiment_type TEXT,
    version TEXT,
    mrr_3 REAL,
    mrr_20 REAL,
    hits_1 REAL,
    hits_10 REAL,
    PRIMARY KEY (system, size, experiment_type, version)
);
