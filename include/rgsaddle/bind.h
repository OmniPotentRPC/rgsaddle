/**
 * @file bind.h
 * @brief Thread-local last-error slot for the Python binding.
 *
 * F2 allows a thread-local last-error string. F7 reads it on the same
 * thread when a status becomes a RuntimeError or NumericalError.
 * This header is not part of the session ABI in rgsaddle.h.
 */
#ifndef RGSADDLE_BIND_H_
#define RGSADDLE_BIND_H_

#ifdef __cplusplus
extern "C" {
#endif

/** Last error text for this thread. Empty when unset. */
const char *rgsaddle_last_error(void);

/** Replace the thread-local text. NULL clears it. */
void rgsaddle_set_last_error(const char *text);

#ifdef __cplusplus
}
#endif

#endif /* RGSADDLE_BIND_H_ */
