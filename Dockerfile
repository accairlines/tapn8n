# production environment
FROM nginx:stable-alpine

# Copy main nginx configuration
COPY ./nginx.conf /etc/nginx/nginx.conf

# Copy SSL certificates
COPY ./ssl /usr/share/ssl

# Copy .htpasswd for /llama authentication
COPY ./.htpasswd /etc/nginx/.htpasswd

# Expose standard HTTP(S) ports
EXPOSE 80 443

# Start nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
