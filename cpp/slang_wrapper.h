#ifndef SLANG_WRAPPER_H
#define SLANG_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Optimization levels (map to SLANG_OPTIMIZATION_LEVEL_*). */
typedef enum {
    SLANGW_OPT_NONE    = 0,
    SLANGW_OPT_DEFAULT = 1,
    SLANGW_OPT_HIGH    = 2,
    SLANGW_OPT_MAXIMAL = 3,
} SlangwOptLevel;

/*
 * Global session — one per process.
 * Returns NULL on failure.
 * Caller must call slangw_global_session_release() at process shutdown.
 */
void* slangw_create_global_session(void);
void  slangw_global_session_release(void* gs);

/*
 * Per-compilation session.
 * Caller owns; must call slangw_release() when done.
 */
void* slangw_create_session(
    void*               global_session,
    const char* const*  search_paths,
    int                 search_path_count,
    SlangwOptLevel      opt_level);

/*
 * Load a module from source and compile all its entry points to SPIR-V
 * in a single call.  The IModule is kept internal to the session and is
 * never exposed across the FFI boundary (avoids ownership ambiguity with
 * the session's internal module cache).
 *
 * session:    an ISession* returned by slangw_create_session().
 * name:       module name (used for diagnostics / caching inside the session).
 * source:     Slang source text (UTF-8).
 * source_len: byte length of source (0 = use strlen).
 * target_idx: target index within the session (pass 0 for the single SPIRV target).
 * out_code:   on success, set to a heap-allocated SPIR-V byte buffer.
 *             Caller must call slangw_free_buffer() when done.
 * out_size:   on success, byte count of *out_code.
 * out_diag:   on failure, set to a NUL-terminated diagnostic string.
 *             Caller must call slangw_free_string() when done.
 *             Set to NULL on success.
 *
 * Returns 0 on success, negative error code on failure.
 */
int slangw_compile_to_spirv(
    void*               session,
    const char*         name,
    const char*         source,
    size_t              source_len,
    int                 target_idx,
    const void**        out_code,
    size_t*             out_size,
    char**              out_diag);

/* Free a SPIR-V buffer returned by slangw_compile_to_spirv. */
void slangw_free_buffer(void* ptr);

/* Free a diagnostic string returned by slangw_compile_to_spirv. */
void slangw_free_string(char* ptr);

/* Release an ISlangUnknown object (e.g. a session). */
void slangw_release(void* obj);

/* Shut down the Slang runtime (call after slangw_global_session_release). */
void slangw_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* SLANG_WRAPPER_H */
