package parser

import (
	"testing"

	"github.com/cavenine/ctxpp/internal/types"
)

func TestSQLParser_Language(t *testing.T) {
	p := NewSQLParser()
	if got := p.Language(); got != "sql" {
		t.Errorf("Language() = %q, want %q", got, "sql")
	}
}

func TestSQLParser_Extensions(t *testing.T) {
	p := NewSQLParser()
	exts := p.Extensions()
	want := map[string]bool{".sql": true}
	for _, ext := range exts {
		if !want[ext] {
			t.Errorf("unexpected extension %q", ext)
		}
		delete(want, ext)
	}
	for ext := range want {
		t.Errorf("missing extension %q", ext)
	}
}

func TestSQLParser_Parse(t *testing.T) {
	p := NewSQLParser()

	tests := []struct {
		name     string
		input    string
		wantSyms []struct {
			name string
			kind types.SymbolKind
		}
	}{
		{
			name: "create table",
			input: `CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "users", kind: types.KindTable},
			},
		},
		{
			name: "create table if not exists",
			input: `CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY
);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "accounts", kind: types.KindTable},
			},
		},
		{
			name: "create view",
			input: `CREATE VIEW active_users AS
SELECT * FROM users WHERE active = true;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "active_users", kind: types.KindView},
			},
		},
		{
			name: "create or replace view",
			input: `CREATE OR REPLACE VIEW mv_data AS
SELECT id, name FROM data;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "mv_data", kind: types.KindView},
			},
		},
		{
			name:  "create index",
			input: `CREATE INDEX idx_users_email ON users (email);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "idx_users_email", kind: types.KindIndex},
			},
		},
		{
			name:  "create unique index",
			input: `CREATE UNIQUE INDEX idx_users_unique_email ON users (email);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "idx_users_unique_email", kind: types.KindIndex},
			},
		},
		{
			name: "create function",
			input: `CREATE FUNCTION get_user(user_id INT)
RETURNS TABLE(id INT, name TEXT)
AS $$
BEGIN
    RETURN QUERY SELECT id, name FROM users WHERE id = user_id;
END;
$$ LANGUAGE plpgsql;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "get_user", kind: types.KindFunction},
			},
		},
		{
			name: "create or replace function",
			input: `CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "update_timestamp", kind: types.KindFunction},
			},
		},
		{
			name: "create procedure",
			input: `CREATE PROCEDURE migrate_data(batch_size INT)
LANGUAGE plpgsql
AS $$
BEGIN
    -- migration logic
END;
$$;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "migrate_data", kind: types.KindProcedure},
			},
		},
		{
			name: "create trigger",
			input: `CREATE TRIGGER update_users_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "update_users_timestamp", kind: types.KindTrigger},
			},
		},
		{
			name: "multiple statements",
			input: `-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

-- Index on name
CREATE INDEX idx_users_name ON users (name);

CREATE VIEW active_users AS
SELECT * FROM users WHERE deleted_at IS NULL;

CREATE FUNCTION count_users()
RETURNS INTEGER AS $$
    SELECT COUNT(*) FROM users;
$$ LANGUAGE sql;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "users", kind: types.KindTable},
				{name: "idx_users_name", kind: types.KindIndex},
				{name: "active_users", kind: types.KindView},
				{name: "count_users", kind: types.KindFunction},
			},
		},
		{
			name: "schema-qualified name",
			input: `CREATE TABLE public.events (
    id SERIAL PRIMARY KEY
);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "events", kind: types.KindTable},
			},
		},
		{
			name: "case insensitive",
			input: `create table Orders (
    id serial primary key
);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "Orders", kind: types.KindTable},
			},
		},
		{
			name: "leading comment captured",
			input: `-- Track user activity across sessions.
-- Records page views and actions.
CREATE TABLE user_activity (
    id SERIAL PRIMARY KEY
);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "user_activity", kind: types.KindTable},
			},
		},
		{
			name:  "empty file",
			input: "",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name:  "comments only",
			input: "-- just a comment\n-- another comment\n",
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{},
		},
		{
			name: "alter and insert ignored",
			input: `ALTER TABLE users ADD COLUMN age INT;
INSERT INTO users (name) VALUES ('alice');
CREATE TABLE logs (id SERIAL);`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "logs", kind: types.KindTable},
			},
		},
		{
			name: "materialized view",
			input: `CREATE MATERIALIZED VIEW mv_daily_stats AS
SELECT date, count(*) FROM events GROUP BY date;`,
			wantSyms: []struct {
				name string
				kind types.SymbolKind
			}{
				{name: "mv_daily_stats", kind: types.KindView},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := p.Parse("test.sql", []byte(tc.input))
			if err != nil {
				t.Fatalf("Parse() error: %v", err)
			}
			if len(result.Symbols) != len(tc.wantSyms) {
				t.Fatalf("got %d symbols, want %d:\n  got:  %v",
					len(result.Symbols), len(tc.wantSyms), symbolNames(result.Symbols))
			}
			for i, want := range tc.wantSyms {
				got := result.Symbols[i]
				if got.Name != want.name {
					t.Errorf("symbol[%d].Name = %q, want %q", i, got.Name, want.name)
				}
				if got.Kind != want.kind {
					t.Errorf("symbol[%d].Kind = %q, want %q", i, got.Kind, want.kind)
				}
				if got.File != "test.sql" {
					t.Errorf("symbol[%d].File = %q, want %q", i, got.File, "test.sql")
				}
				if got.StartLine == 0 {
					t.Errorf("symbol[%d].StartLine = 0, want nonzero", i)
				}
			}
		})
	}
}

func TestSQLParser_LeadingComment(t *testing.T) {
	p := NewSQLParser()
	input := `-- Track user activity across sessions.
-- Records page views and actions.
CREATE TABLE user_activity (
    id SERIAL PRIMARY KEY
);`
	result, err := p.Parse("test.sql", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	sym := result.Symbols[0]
	if sym.DocComment == "" {
		t.Error("expected non-empty DocComment for symbol with leading comments")
	}
}

func TestSQLParser_Signature(t *testing.T) {
	p := NewSQLParser()
	input := `CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);`
	result, err := p.Parse("test.sql", []byte(input))
	if err != nil {
		t.Fatalf("Parse() error: %v", err)
	}
	if len(result.Symbols) != 1 {
		t.Fatalf("got %d symbols, want 1", len(result.Symbols))
	}
	sig := result.Symbols[0].Signature
	if sig == "" {
		t.Error("expected non-empty Signature")
	}
}

func symbolNames(syms []types.Symbol) []string {
	var names []string
	for _, s := range syms {
		names = append(names, s.Name+"("+string(s.Kind)+")")
	}
	return names
}
