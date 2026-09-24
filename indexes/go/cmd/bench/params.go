package main

import (
	"math"
	"strconv"
	"strings"
)

// Parameter values are int64, float64 or string. The type of the default
// decides the type of the value.

// parseValue converts a command-line value to the type of def.
func parseValue(key, raw string, def any) (any, error) {
	switch def.(type) {
	case int64:
		v, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			f, ferr := strconv.ParseFloat(raw, 64)
			if ferr != nil || f != math.Trunc(f) {
				return nil, usagef("parameter %s needs an integer, got %q", key, raw)
			}
			v = int64(f)
		}
		return v, nil
	case float64:
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return nil, usagef("parameter %s needs a number, got %q", key, raw)
		}
		return v, nil
	default:
		return raw, nil
	}
}

// resolveParams copies defaults and applies KEY=VALUE items. Each item may
// hold several comma-separated pairs.
func resolveParams(defaults map[string]any, items []string, phase string) (map[string]any, error) {
	p := make(map[string]any, len(defaults))
	for k, v := range defaults {
		p[k] = v
	}
	for _, item := range items {
		for _, pair := range strings.Split(item, ",") {
			key, raw, ok := strings.Cut(strings.TrimSpace(pair), "=")
			if !ok || key == "" {
				return nil, usagef("bad %s parameter %q, want KEY=VALUE", phase, pair)
			}
			def, known := defaults[key]
			if !known {
				return nil, usagef("unknown %s parameter %q", phase, key)
			}
			v, err := parseValue(key, raw, def)
			if err != nil {
				return nil, err
			}
			p[key] = v
		}
	}
	return p, checkEnums(p)
}

// checkEnums rejects string values outside the sets of section 6.
func checkEnums(p map[string]any) error {
	allowed := map[string][]string{"metric": {"ip", "l2"}, "io": {"mmap", "nocache"}}
	for key, vals := range allowed {
		v, ok := p[key].(string)
		if !ok {
			continue
		}
		found := false
		for _, a := range vals {
			found = found || v == a
		}
		if !found {
			return usagef("parameter %s must be one of %v, got %q", key, vals, v)
		}
	}
	return nil
}

// searchParamSets returns one resolved map per --search flag, or the defaults
// once if no --search was given.
func searchParamSets(defaults map[string]any, items []string) ([]map[string]any, error) {
	if len(items) == 0 {
		p, err := resolveParams(defaults, nil, "search")
		return []map[string]any{p}, err
	}
	sets := make([]map[string]any, 0, len(items))
	for _, item := range items {
		p, err := resolveParams(defaults, []string{item}, "search")
		if err != nil {
			return nil, err
		}
		sets = append(sets, p)
	}
	return sets, nil
}
