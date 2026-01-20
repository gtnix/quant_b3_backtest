/**
 * Centralized API Helper
 * 
 * Ensures all API calls use the correct base URL for both Tauri and Browser modes.
 * Use this instead of raw fetch() for API calls.
 */

import { config } from './platform';

type RequestOptions = Omit<RequestInit, 'body'> & {
  body?: object | string;
};

/**
 * Make an API request with automatic base URL handling
 */
async function request<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { body, ...restOptions } = options;
  
  const url = `${config.apiBase}${endpoint}`;
  
  const fetchOptions: RequestInit = {
    ...restOptions,
    headers: {
      'Content-Type': 'application/json',
      ...restOptions.headers,
    },
  };
  
  if (body) {
    fetchOptions.body = typeof body === 'string' ? body : JSON.stringify(body);
  }
  
  const response = await fetch(url, fetchOptions);
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `API error: ${response.status}`);
  }
  
  return response.json();
}

export const api = {
  /**
   * GET request
   */
  get: <T>(endpoint: string, options?: RequestOptions) => 
    request<T>(endpoint, { ...options, method: 'GET' }),
  
  /**
   * POST request
   */
  post: <T>(endpoint: string, body?: object, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: 'POST', body }),
  
  /**
   * PATCH request
   */
  patch: <T>(endpoint: string, body?: object, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: 'PATCH', body }),
  
  /**
   * PUT request
   */
  put: <T>(endpoint: string, body?: object, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: 'PUT', body }),
  
  /**
   * DELETE request
   */
  delete: <T>(endpoint: string, options?: RequestOptions) =>
    request<T>(endpoint, { ...options, method: 'DELETE' }),
};

export default api;
