import { serialize } from 'cookie';

export default function handler(req, res) {
  // Clear the auth_token cookie
  const cookie = serialize('auth_token', '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    expires: new Date(0), // Expire immediately
    path: '/',
  });

  res.setHeader('Set-Cookie', cookie);
  
  // Clean JSON response or redirection can be done in client, but redirect is elegant
  res.writeHead(302, { Location: '/' });
  res.end();
}
