import { NextResponse } from 'next/server';

export function middleware(request) {
  const { pathname } = request.nextUrl;

  // Protect both /dashboard and /dashboard.html
  if (pathname === '/dashboard' || pathname.startsWith('/dashboard.html')) {
    const token = request.cookies.get('auth_token')?.value;

    if (!token || token !== 'authenticated') {
      const url = request.nextUrl.clone();
      url.pathname = '/';
      return NextResponse.redirect(url);
    }

    // Rewrite /dashboard to /dashboard.html so they get served the static file directly and cleanly
    if (pathname === '/dashboard') {
      const url = request.nextUrl.clone();
      url.pathname = '/dashboard.html';
      return NextResponse.rewrite(url);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard', '/dashboard.html'],
};
