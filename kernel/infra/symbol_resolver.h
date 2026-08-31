#ifndef __KSU_SYMBOL_RESOLVER_H
#define __KSU_SYMBOL_RESOLVER_H

/*
 * Compatibility shim for the symbol resolver API used by upstream (JingMatrix)
 * features: find_kernel_symbol_exact() / ksu_resolve_symbol_for_functable_hook()
 * / ksu_init_symbol_resolver().
 *
 * This tree already carries a more thorough resolver in
 * downstream/kallsyms_common.h, which walks through
 *   kallsyms_lookup_name -> kprobes -> kallsyms_on_each_symbol -> hashed array
 * and additionally covers legacy/downstream kernels. Map the upstream API onto
 * it so imported features build unmodified instead of duplicating the logic.
 */

#ifdef CONFIG_KALLSYMS

static inline unsigned long find_kernel_symbol_exact(const char *symbol_name)
{
	return (unsigned long)kallsyms_lookup_retry(symbol_name);
}

static inline void *ksu_resolve_symbol_for_functable_hook(const char *symbol_name)
{
	return (void *)kallsyms_lookup_retry(symbol_name);
}

#else

static inline unsigned long find_kernel_symbol_exact(const char *symbol_name)
{
	(void)symbol_name;
	return 0;
}

static inline void *ksu_resolve_symbol_for_functable_hook(const char *symbol_name)
{
	(void)symbol_name;
	return NULL;
}

#endif

/*
 * This tree builds its kallsyms hash array lazily on first miss, so there is
 * nothing to prime at init time.
 */
static inline void ksu_init_symbol_resolver(void)
{
}

#endif
